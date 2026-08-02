import json

with open("nghi_dinh_168_parsed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("======================================================================")
print(" 1. QUÉT TOÀN BỘ FILE JSON TÌM CHỮ 'DÂY' HOẶC 'THẮT'")
print("======================================================================\n")

found_count = 0
for chunk in data:
    text = chunk.get("full_legal_text", "").lower()
    if "dây" in text or "thắt" in text:
        print(f"📌 ID: {chunk['id']}")
        print(f"   Article Title: {chunk.get('article', '')}")
        print(f"   Text: {chunk.get('full_legal_text', '')[:150]}...\n" + "-"*60)
        found_count += 1

if found_count == 0:
    print("❌ BÁO ĐỘNG: Trong TOÀN BỘ file JSON KHÔNG HỆ CÓ từ 'dây' hoặc 'thắt'!")
    print("   -> File nghi_dinh_168_parsed.json từ Step 1 đã bị parser làm sót/mất đoạn này.\n")

print("\n======================================================================")
print(" 2. MỤC LỤC TẤT CẢ CÁC ĐIỀU TRONG FILE JSON CỦA BẠN")
print("======================================================================\n")

articles = {}
for chunk in data:
    chunk_id = chunk.get("id", "")
    art_title = chunk.get("article", "")
    dieu_id = chunk_id.split("_K")[0].split("_D")[0]
    if dieu_id not in articles:
        articles[dieu_id] = art_title

# In danh sách tất cả các Điều có trong file
for dieu, title in sorted(articles.items()):
    print(f"• [{dieu}]: {title[:80]}")