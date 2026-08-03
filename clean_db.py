import psycopg2

POSTGRES_CONFIG = {
    "dbname": "traffic_db",
    "user": "postgres",
    "password": "", # Thay mật khẩu Postgres của bạn
    "host": "localhost",
    "port": "5432"
}

def reset_database():
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()
        
        # Xóa bảng cũ
        cur.execute("DROP TABLE IF EXISTS legal_provisions CASCADE;")
        conn.commit()
        
        cur.close()
        conn.close()
        print("-> [Thành công] Đã xóa sạch bảng 'legal_provisions' trong PostgreSQL!")
    except Exception as e:
        print(f"-> [Lỗi]: {e}")

if __name__ == "__main__":
    reset_database()