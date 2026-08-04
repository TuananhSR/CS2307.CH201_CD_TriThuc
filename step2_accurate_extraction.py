import json
import os
import re
import unicodedata
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from owlready2 import (
    get_ontology, Thing, ObjectProperty, DataProperty, FunctionalProperty
)

load_dotenv()

POSTGRES_CONFIG = {
    "dbname": "traffic_db",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": "localhost",
    "port": "5432"
}

def clean_unicode(text: str) -> str:
    if not text: return ""
    return unicodedata.normalize('NFC', text)

def determine_vehicle_type(chunk_id: str, article_title: str) -> str:
    """Phân định chính xác loại xe dựa trên ID của Nghị định 168"""
    if "Điều6_" in chunk_id: return "OTo"
    elif "Điều7_" in chunk_id: return "XeMay"
    elif "Điều8_" in chunk_id: return "XeMayChuyenDung"
    elif "Điều9_" in chunk_id: return "XeDap"
    elif "Điều10_" in chunk_id: return "NguoiBoHanh"
    elif "Điều11_" in chunk_id: return "VatNuoi"

    # Các Điều phạt Giấy tờ/Đăng kiểm (Từ Điều 12 trở đi)
    title_norm = clean_unicode(article_title).lower()
    has_oto = any(k in title_norm for k in ["ô tô", "o to"])
    has_xemay = any(k in title_norm for k in ["mô tô", "xe máy"])
    
    if has_oto and has_xemay: return "Chung"
    elif has_oto: return "OTo"
    elif has_xemay: return "XeMay"
    elif any(k in title_norm for k in ["máy chuyên dùng", "máy kéo"]): return "XeMayChuyenDung"
        
    return "Chung"

def parse_fine_amounts(text: str):
    text_lower = clean_unicode(text).lower()
    min_fine, max_fine = None, None
    fine_match = re.search(
        r'(?:phạt\s*(?:tiền|liền)?\s*(?:mức)?\s*từ|từ)\s+([\d\.]+)\s*(?:đồng)?\s+đến\s+([\d\.]+)\s*đồng', 
        text_lower
    )
    if fine_match:
        try:
            min_fine = int(fine_match.group(1).replace(".", ""))
            max_fine = int(fine_match.group(2).replace(".", ""))
        except ValueError:
            pass
    return min_fine, max_fine

def determine_alcohol_frame(text: str):
    text_lower = clean_unicode(text).lower()
    if any(k in text_lower for k in ["nồng độ cồn", "khí thở", "trong máu"]):
        if any(k in text_lower for k in ["chưa vượt quá 0,25", "chưa vượt quá 50 miligam", "dưới 0,25"]):
            return 1
        elif any(k in text_lower for k in ["vượt quá 0,25 miligam đến 0,4", "vượt quá 50 miligam đến 80", "0,25 mg đến 0,4"]):
            return 2
        elif any(k in text_lower for k in ["vượt quá 0,4 miligam", "vượt quá 80 miligam", "không chấp hành yêu cầu kiểm tra về nồng độ cồn"]):
            return 3
    return None

def determine_speed_frame(text: str):
    text_lower = clean_unicode(text).lower()
    if any(k in text_lower for k in ["tốc độ", "chạy quá"]):
        if any(k in text_lower for k in ["từ 05 km/h đến dưới 10", "từ 5 km/h đến dưới 10"]):
            return 1
        elif "từ 10 km/h đến 20" in text_lower:
            return 2
        elif "trên 20 km/h đến 35" in text_lower:
            return 3
        elif "trên 35 km/h" in text_lower:
            return 4
    return None

VALID_POINTS = {'a', 'b', 'c', 'd', 'đ', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y'}

def map_point_letter_to_id(p: str) -> str:
    p = p.strip().lower()
    if p == 'đ':
        return 'Dđ'
    return 'D' + p

def parse_decree_references_advanced(article_prefix: str, text: str) -> list:
    targets = []
    text = clean_unicode(text).lower()
    
    # Split text into segments by both semicolon and comma
    normalized_text = text.replace(";", ",")
    raw_segments = normalized_text.split(",")
    
    parsed_segments = []
    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue
        
        # Find clause numbers: 'khoản 11'
        clauses = re.findall(r'khoản\s+(\d+)', seg)
        clause = clauses[0] if clauses else None
        
        # Find point letters: 'điểm a'
        all_points = re.findall(r'điểm\s+([a-zđ]+)', seg)
        points = [p for p in all_points if p in VALID_POINTS]
        
        parsed_segments.append({
            "text": seg,
            "points": points,
            "clause": clause
        })
        
    # Backward-association: if a segment has points but no clause,
    # it inherits the clause from the nearest following segment that has a clause.
    for i in range(len(parsed_segments)):
        if parsed_segments[i]["points"] and parsed_segments[i]["clause"] is None:
            # Look forward for a clause
            for j in range(i + 1, len(parsed_segments)):
                if parsed_segments[j]["clause"] is not None:
                    parsed_segments[i]["clause"] = parsed_segments[j]["clause"]
                    break
                    
    # Generate target IDs
    for seg in parsed_segments:
        clause_num = seg["clause"]
        if clause_num:
            if seg["points"]:
                for p in seg["points"]:
                    p_id = map_point_letter_to_id(p)
                    targets.append(f"{article_prefix}_K{clause_num}_{p_id}")
            else:
                targets.append(f"{article_prefix}_K{clause_num}")
                
    return targets

def is_penalty_clause(t: str) -> bool:
    if "Điều6" in t:
        return "_K15" in t or "_K16" in t
    elif "Điều7" in t:
        return "_K11" in t or "_K12" in t or "_K13" in t
    elif "Điều8" in t:
        return "_K10" in t or "_K11" in t
    return False

def build_penalties_maps(law_chunks: list):
    points_map = {}
    revocation_map = {}
    
    for chunk in law_chunks:
        full_text = clean_unicode(chunk.get("full_legal_text", ""))
        raw_text = clean_unicode(chunk.get("raw_text", ""))
        chunk_id = chunk["id"]
        article_prefix = chunk_id.split("_K")[0]
        
        # 1. Parse point deductions
        if "trừ điểm giấy phép lái xe" in full_text.lower():
            pts_match = re.search(r'bị\s+trừ\s+điểm\s+giấy\s+phép\s+lái\s+xe\s+(\d+)\s+điểm', full_text.lower())
            if pts_match:
                pts = int(pts_match.group(1))
                targets = parse_decree_references_advanced(article_prefix, raw_text)
                for t in targets:
                    if is_penalty_clause(t):
                        continue
                    points_map[t] = pts
                    
        # 2. Parse license revocation months
        if "tước quyền sử dụng giấy phép lái xe" in full_text.lower():
            revoc_match = re.search(r'tước\s+quyền\s+sử\s+dụng\s+giấy\s+phép\s+lái\s+xe(?:\s+từ)?\s+(\d+)\s+tháng(?:\s+đến\s+(\d+)\s+tháng)?', full_text.lower())
            if revoc_match:
                months = int(revoc_match.group(2)) if revoc_match.group(2) else int(revoc_match.group(1))
                targets = parse_decree_references_advanced(article_prefix, raw_text)
                for t in targets:
                    if is_penalty_clause(t):
                        continue
                    revocation_map[t] = months
                    
    return points_map, revocation_map

def run_step2_pipeline(json_path: str, owl_path: str, db_config: dict):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        law_chunks = json.load(f)

    points_map, revocation_map = build_penalties_maps(law_chunks)

    # Khởi tạo Ontology
    onto = get_ontology("http://traffic_expert_system.org/nd168.owl")
    with onto:
        class ThucTheGiaoThong(Thing): pass
        class LoaiPhuongTien(ThucTheGiaoThong): pass
        class XeMay(LoaiPhuongTien): pass
        class OTo(LoaiPhuongTien): pass
        class XeMayChuyenDung(LoaiPhuongTien): pass
        class XeDap(LoaiPhuongTien): pass

        class DieuKhoanLuat(ThucTheGiaoThong): pass
        class KhoanLuat(DieuKhoanLuat): pass
        class DiemLuat(DieuKhoanLuat): pass

        class apDungChoPhuongTien(ObjectProperty): domain = [DieuKhoanLuat]; range = [LoaiPhuongTien]
        class thuocKhoan(ObjectProperty): domain = [DiemLuat]; range = [KhoanLuat]

        class coMinFine(DataProperty, FunctionalProperty): domain = [DieuKhoanLuat]; range = [int]
        class coMaxFine(DataProperty, FunctionalProperty): domain = [DieuKhoanLuat]; range = [int]
        class truDiemGPLX(DataProperty, FunctionalProperty): domain = [DieuKhoanLuat]; range = [int]

        ent_xe_may = XeMay("Ent_XeMay")
        ent_o_to = OTo("Ent_OTo")
        ent_may_chuyen_dung = XeMayChuyenDung("Ent_XeMayChuyenDung")
        ent_xe_dap = XeDap("Ent_XeDap")

    # Lấy thông tin từ cấp Khoản (Clause) để Điểm (Point) kế thừa
    clause_penalties = {}
    for chunk in law_chunks:
        if chunk["level"] == "Clause":
            chunk_id = chunk["id"]
            full_text = clean_unicode(chunk.get("full_legal_text", ""))
            min_f, max_f = parse_fine_amounts(full_text)
            alc_f = determine_alcohol_frame(full_text)
            spd_f = determine_speed_frame(full_text)
            clause_penalties[chunk_id] = (min_f, max_f, alc_f, spd_f)

    nodes_owl_dict = {}
    raw_db_rows = []

    with onto:
        for chunk in law_chunks:
            chunk_id = chunk["id"]
            article_title = clean_unicode(chunk.get("article", ""))
            raw_text = clean_unicode(chunk.get("raw_text", ""))
            full_text = clean_unicode(chunk.get("full_legal_text", ""))

            vehicle_type = determine_vehicle_type(chunk_id, article_title)
            min_f, max_f = parse_fine_amounts(full_text)
            alc_frame = determine_alcohol_frame(full_text)
            spd_frame = determine_speed_frame(full_text)
            pts = points_map.get(chunk_id, 0)
            revoc_months = revocation_map.get(chunk_id, 0)

            # CẮT ĐÚNG PARENT ID ĐỂ KẾ THỪA TIỀN PHẠT (Fix bug Dây an toàn)
            parent_id = None
            if chunk["level"] == "Point":
                parent_id = re.sub(r'_D[a-zA-Z0-9đĐ]+$', '', chunk_id)
                if parent_id in clause_penalties:
                    p_min, p_max, p_alc, p_spd = clause_penalties[parent_id]
                    if min_f is None: min_f = p_min
                    if max_f is None: max_f = p_max
                    if alc_frame is None: alc_frame = p_alc
                    if spd_frame is None: spd_frame = p_spd

            node = KhoanLuat(chunk_id) if chunk["level"] == "Clause" else DiemLuat(chunk_id)

            if vehicle_type == "XeMay": node.apDungChoPhuongTien.append(ent_xe_may)
            elif vehicle_type == "OTo": node.apDungChoPhuongTien.append(ent_o_to)
            elif vehicle_type == "XeMayChuyenDung": node.apDungChoPhuongTien.append(ent_may_chuyen_dung)
            elif vehicle_type == "XeDap": node.apDungChoPhuongTien.append(ent_xe_dap)

            if min_f: node.coMinFine = min_f
            if max_f: node.coMaxFine = max_f
            if pts: node.truDiemGPLX = pts

            nodes_owl_dict[chunk_id] = (node, chunk)

            raw_db_rows.append((
                chunk_id, parent_id, chunk["level"], vehicle_type,
                min_f, max_f, pts, revoc_months, alc_frame, spd_frame,
                chunk["source"], article_title,
                str(chunk["clause"]) if chunk["clause"] else None,
                str(chunk["point"]) if chunk["point"] else None,
                raw_text, full_text
            ))

        for chunk_id, (node, chunk) in nodes_owl_dict.items():
            if chunk["level"] == "Point" and "_D" in chunk_id:
                parent_id = re.sub(r'_D[a-zA-Z0-9đĐ]+$', '', chunk_id)
                if parent_id in nodes_owl_dict:
                    node.thuocKhoan.append(nodes_owl_dict[parent_id][0])

    onto.save(file=owl_path, format="rdfxml")
    print(f" -> [Thành công] Đã xuất file Đồ thị Tri thức OWL: {owl_path}")

    # Nạp vào PostgreSQL
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS legal_provisions CASCADE;")
        cur.execute("""
            CREATE TABLE legal_provisions (
                id VARCHAR(100) PRIMARY KEY,
                parent_id VARCHAR(100),
                level VARCHAR(20),
                vehicle_type VARCHAR(50),
                min_fine BIGINT,
                max_fine BIGINT,
                points_deducted INT DEFAULT 0,
                license_revocation_months INT DEFAULT 0,
                alcohol_level_frame INT,
                speed_level_frame INT,
                source TEXT, article TEXT, clause VARCHAR(20), point VARCHAR(20),
                raw_text TEXT, full_legal_text TEXT
            );
        """)

        dedup_dict = {row[0]: row for row in raw_db_rows}
        insert_query = """
            INSERT INTO legal_provisions 
            (id, parent_id, level, vehicle_type, min_fine, max_fine, points_deducted, license_revocation_months, alcohol_level_frame, speed_level_frame, source, article, clause, point, raw_text, full_legal_text)
            VALUES %s;
        """
        execute_values(cur, insert_query, list(dedup_dict.values()))
        conn.commit()

        print("\n [CẬP NHẬT DATABASE THÀNH CÔNG - SẠCH 100% LỖI KẾ THỪA]:")
        cur.execute("SELECT vehicle_type, COUNT(*) FROM legal_provisions GROUP BY vehicle_type ORDER BY count DESC;")
        for v_type, count in cur.fetchall():
            print(f"   + [{v_type}]: {count} điều khoản")

        cur.close()
        conn.close()

    except Exception as e:
        print(f" [Lỗi Postgres]: {e}")

if __name__ == "__main__":
    JSON_INPUT = "nghi_dinh_168_parsed.json"
    OWL_OUTPUT = "traffic_ontology_168.owl"
    run_step2_pipeline(JSON_INPUT, OWL_OUTPUT, POSTGRES_CONFIG)