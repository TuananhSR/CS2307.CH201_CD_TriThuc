import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

POSTGRES_CONFIG = {
    "dbname": "traffic_db",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": "localhost",
    "port": "5432"
}

def check_seatbelt_clause():
    conn = psycopg2.connect(**POSTGRES_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()

    print("======================================================================")
    print("      KIỂM TRA DỮ LIỆU ĐIỀU KHOẢN 'DÂY AN TOÀN' TRONG POSTGRESQL")
    print("======================================================================\n")

    cur.execute("SELECT id, vehicle_type, min_fine, max_fine, raw_text FROM legal_provisions WHERE full_legal_text ILIKE '%dây an toàn%';")
    rows = cur.fetchall()

    if not rows:
        print("❌ [LỖI 1]: KHÔNG tìm thấy bất kỳ bản ghi nào chứa từ 'dây an toàn'!")
    else:
        print(f"✅ Tìm thấy {len(rows)} bản ghi chứa cụm từ 'dây an toàn':\n")
        for r in rows:
            print(f"📌 ID: {r['id']}")
            print(f"   • vehicle_type : '{r['vehicle_type']}'")
            print(f"   • min_fine     : {r['min_fine']}")
            print(f"   • max_fine     : {r['max_fine']}")
            print(f"   • raw_text     : {r['raw_text'][:120]}...\n")

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_seatbelt_clause()