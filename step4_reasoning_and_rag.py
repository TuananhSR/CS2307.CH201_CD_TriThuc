import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from step3_event_extraction import EventExtractor, SuKienGiaoThong

load_dotenv()

POSTGRES_CONFIG = {
    "dbname": "traffic_db",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": "localhost",
    "port": "5432"
}

class UniversalLegalReasoningEngine:
    def __init__(self, db_config: dict):
        self.db_config = db_config

    def get_db_connection(self):
        return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)

    def _determine_alc_frame(self, alc_val: float) -> int:
        if alc_val <= 0.25: return 1
        elif 0.25 < alc_val <= 0.4: return 2
        else: return 3

    def _determine_speed_frame(self, spd_val: float) -> int:
        if 5 <= spd_val <= 10: return 1
        elif 10 < spd_val <= 20: return 2
        elif 20 < spd_val <= 35: return 3
        else: return 4

    def _search_single_scenario(self, cur, event: SuKienGiaoThong) -> list:
        matched_dict = {}
        alcohol_bracket_items = []

        # A. XỬ LÝ CHỦ PHƯƠNG TIỆN
        if event.doi_tuong_bi_xu_phat == "ChuPhuongTien":
            for hv in event.danh_sach_hanh_vi:
                sql = "SELECT * FROM legal_provisions WHERE id LIKE 'ND168_Điều32%%' AND full_legal_text ILIKE %s AND min_fine > 0 LIMIT 1;"
                cur.execute(sql, (f"%{hv.mo_ta}%",))
                r = cur.fetchone()
                if r: matched_dict[r["id"]] = r
            return list(matched_dict.values()), []

        # B. XỬ LÝ LỖI NGƯỜI ĐIỀU KHIỂN
        for hv in event.danh_sach_hanh_vi:
            code = hv.ma_hanh_vi

            if code == "NONG_DO_CON":
                if hv.chi_so_dinh_luong is not None:
                    alc_frame = self._determine_alc_frame(hv.chi_so_dinh_luong)
                    sql = "SELECT * FROM legal_provisions WHERE vehicle_type = %s AND alcohol_level_frame = %s AND min_fine > 0 LIMIT 1;"
                    cur.execute(sql, (event.loai_phuong_tien, alc_frame))
                    r = cur.fetchone()
                    if r: matched_dict[r["id"]] = r
                else:
                    sql = "SELECT * FROM legal_provisions WHERE vehicle_type = %s AND alcohol_level_frame IS NOT NULL AND min_fine > 0 ORDER BY alcohol_level_frame ASC;"
                    cur.execute(sql, (event.loai_phuong_tien,))
                    alcohol_bracket_items = cur.fetchall()

            elif code == "TOC_DO" and hv.chi_so_dinh_luong is not None:
                spd_frame = self._determine_speed_frame(hv.chi_so_dinh_luong)
                sql = "SELECT * FROM legal_provisions WHERE vehicle_type = %s AND speed_level_frame = %s AND min_fine > 0 LIMIT 1;"
                cur.execute(sql, (event.loai_phuong_tien, spd_frame))
                r = cur.fetchone()
                if r: matched_dict[r["id"]] = r

            else:
                keyword_map = {
                    "KHONG_CHAP_HANH_CSGT": "%người điều khiển giao thông%",
                    "VUOT_DEN_DO": "%không chấp hành%đèn tín hiệu%",
                    "QUEN_GPLX": "%không mang%giấy phép lái xe%",
                    "KHONG_GPLX": "%không có giấy phép lái xe%",
                    "DANG_KIEM_HET_HAN_DUEI_1T": "%dưới 01 tháng%",
                    "KHONG_DONG_MU_BH": "%mũ bảo hiểm%",
                    "KHONG_THAT_DAY_AN_TOAN": "%dây%an toàn%",
                    "DUNG_DIEN_THOAI": "%điện thoại%",
                    "GAY_TAI_NAN_BO_CHAY": "%gây tai nạn%không dừng%",
                    "LANG_LACH_DANH_VONG": "%lạng lách%đánh võng%",
                    "DI_NGUOC_CHIEU_DUONG_CAM": "%ngược chiều%",
                    "MA_TUY": "%ma túy%",
                    "DUNG_DO_SAI_QUY_DINH": "%dừng xe%đỗ xe%",
                    "KHONG_BAT_DEN": "%đèn chiếu sáng%",
                    "CHUYEN_HUONG_KHONG_XI_NHAN": "%tín hiệu báo hướng%",
                    "DUA_XE_TRAI_PHEP": "%đua xe%",
                    "CHO_QUA_SO_NGUOI": "%được phép chở%",
                    "CHO_QUA_TAI": "%tải trọng%"
                }
                pattern = keyword_map.get(code, f"%{hv.mo_ta}%")
                
                sql = "SELECT * FROM legal_provisions WHERE (vehicle_type = %s OR vehicle_type = 'Chung') AND full_legal_text ILIKE %s AND min_fine > 0;"
                cur.execute(sql, (event.loai_phuong_tien, pattern))
                rows = cur.fetchall()

                # BỘ LỌC KHỬ NHIỄM CHÉO & XẾP HẠNG TRÙNG LẶP TỪ KHÓA TỐT NHẤT (Best Match Ranking)
                import re
                filtered_rows = []
                for r in rows:
                    text_lower = r["raw_text"].lower()
                    text_check = text_lower.replace("mô tô", "").replace("moto", "")
                    
                    has_oto = "ô tô" in text_check or "o to" in text_check or "xe hơi" in text_check
                    has_xemay = "mô tô" in text_lower or "xe máy" in text_lower or "xe gắn máy" in text_lower
                    
                    if event.loai_phuong_tien == "OTo":
                        if has_xemay and not has_oto: continue
                    elif event.loai_phuong_tien == "XeMay":
                        if has_oto and not has_xemay: continue
                            
                    filtered_rows.append(r)

                if filtered_rows:
                    desc_tokens = set(re.sub(r'[^\w\s]', ' ', hv.mo_ta.lower()).split())
                    best_row = None
                    max_overlap = -1
                    
                    for r in filtered_rows:
                        r_tokens = set(re.sub(r'[^\w\s]', ' ', r["raw_text"].lower()).split())
                        overlap = len(desc_tokens.intersection(r_tokens))
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_row = r
                    
                    if best_row:
                        matched_dict[best_row["id"]] = best_row

        return list(matched_dict.values()), alcohol_bracket_items

    def _calculate_single_decision(self, event: SuKienGiaoThong, provisions: list, alcohol_brackets: list) -> dict:
        reasoning_trace = []
        detailed_penalties = []
        
        total_min, total_max, total_calc, total_points, max_revocation_months = 0, 0, 0, 0, 0
        has_aggravating = len(event.tinh_tiet_tang_nang) > 0
        has_mitigating = len(event.tinh_tiet_giam_nhe) > 0

        action_descriptions = [hv.mo_ta for hv in event.danh_sach_hanh_vi]
        reasoning_trace.append(f"1. Xác định đối tượng bị xử phạt: [{event.doi_tuong_bi_xu_phat}] điều khiển [{event.loai_phuong_tien}]")
        reasoning_trace.append(f"2. Danh sách hành vi vi phạm: {', '.join(action_descriptions)}")

        for p in provisions:
            min_f = p["min_fine"] or 0
            max_f = p["max_fine"] or 0
            avg_f = (min_f + max_f) // 2

            calc_fine = max_f if (has_aggravating and not has_mitigating) else (min_f if (has_mitigating and not has_aggravating) else avg_f)
            explanation = f"Mức phạt tiền: {calc_fine:,} VNĐ"

            total_min += min_f
            total_max += max_f
            total_calc += calc_fine
            total_points += (p["points_deducted"] or 0)
            
            rev_m = p.get("license_revocation_months", 0) or 0
            if rev_m > max_revocation_months: max_revocation_months = rev_m

            detailed_penalties.append({
                "provision_id": p["id"],
                "article": p["article"],
                "raw_text": p["raw_text"],
                "fine_range": f"{min_f:,} - {max_f:,} VNĐ",
                "calculated_fine": calc_fine,
                "points_deducted": p["points_deducted"],
                "explanation": explanation
            })
            reasoning_trace.append(f"3. Căn cứ [{p['id']}] ({p['article']}): {explanation}")

        alcohol_options_notice = []
        if alcohol_brackets:
            reasoning_trace.append(f"4. Chưa có chỉ số mg/L cụ thể, tổng hợp 3 kịch bản cồn cho [{event.loai_phuong_tien}]:")
            for b in alcohol_brackets:
                min_f = b["min_fine"] or 0
                max_f = b["max_fine"] or 0
                calc_f = max_f if (has_aggravating and not has_mitigating) else ((min_f + max_f) // 2)
                
                tot_revoc = max(max_revocation_months, b.get("license_revocation_months", 0) or 0)
                alcohol_options_notice.append({
                    "alcohol_frame": f"Khung {b['alcohol_level_frame']}",
                    "raw_text": b["raw_text"],
                    "alcohol_fine_range": f"{min_f:,} - {max_f:,} VNĐ",
                    "alcohol_calculated_fine": calc_f,
                    "total_fine_range_including_other_violations": f"{(total_min + min_f):,} - {(total_max + max_f):,} VNĐ",
                    "total_calculated_fine_including_other_violations": total_calc + calc_f,
                    "total_points_deducted": total_points + (b["points_deducted"] or 0),
                    "total_license_revocation_months": tot_revoc
                })

        res = {
            "mode": "SINGLE_SCENARIO_DECISION" if not alcohol_brackets else "SINGLE_VEHICLE_ALCOHOL_BRACKETS",
            "summary": {
                "vehicle_type": event.loai_phuong_tien,
                "target_role": event.doi_tuong_bi_xu_phat,
                "total_calculated_fixed_fine_vnd": total_calc,
                "total_points_deducted_fixed": total_points,
                "max_license_revocation_months": max_revocation_months
            },
            "reasoning_trace": reasoning_trace,
            "detailed_penalties": detailed_penalties
        }

        if alcohol_options_notice:
            res["alcohol_scenarios_for_vehicle"] = alcohol_options_notice

        return res

    def evaluate(self, event: SuKienGiaoThong) -> dict:
        conn = self.get_db_connection()
        cur = conn.cursor()
        try:
            provisions, alcohol_brackets = self._search_single_scenario(cur, event)
            return self._calculate_single_decision(event, provisions, alcohol_brackets)
        finally:
            cur.close()
            conn.close()

class TrafficExpertSystemPipeline:
    def __init__(self, gemini_api_key: str, db_config: dict):
        self.extractor = EventExtractor(api_key=gemini_api_key)
        self.reasoning_engine = UniversalLegalReasoningEngine(db_config=db_config)

    def process(self, user_query: str):
        print("\n" + "="*75)
        print(f" CÂU HỎI NGƯỜI DÙNG: '{user_query}'")
        print("="*75)

        event = self.extractor.extract_event_from_query(user_query)
        decision = self.reasoning_engine.evaluate(event)

        print("\n [KẾT QUẢ SUY LUẬN & ĐÁP ÁN XỬ PHẠT]:")
        print(json.dumps(decision, ensure_ascii=False, indent=4))
        return decision

if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("LỖI: Chưa cấu hình GEMINI_API_KEY trong file .env!")
    else:
        pipeline = TrafficExpertSystemPipeline(
            gemini_api_key=GEMINI_API_KEY, 
            db_config=POSTGRES_CONFIG
        )

        # pipeline.process("Đi ô tô không thắt dây an toàn bị phạt mấy trăm?")
        # pipeline.process("Đi xe máy vượt đèn đỏ, không đội mũ bảo hiểm, uống rượu bia bị phạt thế nào?")
        # pipeline.process("Tôi quên mang giấy phép lái ô tô thì phat bao nhiêu tiền và bị trừ bao nhiêu điểm?")
        pipeline.process("hôm qua tôi chạy oto nhưng lỡ vượt đèn đò, lúc đó có uống rượu, sau khi vượt đèn thì tôi tông trúng một người, tôi sợ quá tôi bỏ chạy, thì bị xử phạt thế nào?")