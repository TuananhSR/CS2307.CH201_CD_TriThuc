import os
import json
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from step3_event_extraction import SuKienGiaoThong
from step4_reasoning_and_rag import TrafficExpertSystemPipeline, POSTGRES_CONFIG

load_dotenv()

class ConversationalTrafficAgent:
    def __init__(self, gemini_api_key: str, db_config: dict):
        self.gemini_api_key = gemini_api_key
        self.db_config = db_config
        
        # Khởi tạo pipeline hệ chuyên gia hiện tại (Step 3 + Step 4)
        self.pipeline = TrafficExpertSystemPipeline(
            gemini_api_key=self.gemini_api_key,
            db_config=self.db_config
        )
        
        # Khởi tạo mô hình Chat của LangChain sử dụng gemini-3.1-flash-lite để đồng bộ
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            api_key=self.gemini_api_key,
            temperature=0.0
        )
        
        # Bộ nhớ lưu trữ lịch sử hội thoại dạng list chứa các HumanMessage và AIMessage
        # Giới hạn tối đa 5 lượt (10 tin nhắn)
        self.chat_history: List[BaseMessage] = []

        # Khởi tạo các Chain thông qua Prompt Templates
        self._init_chains()

    def _init_chains(self):
        # 1. Prompt để Ngữ cảnh hóa câu hỏi (Contextualization Prompt)
        contextualize_system_prompt = (
            "Bạn là trợ lý ảo phân tích hội thoại Luật Giao thông Việt Nam.\n"
            "Dựa vào lịch sử hội thoại dưới đây và một câu hỏi mới nhất từ người dùng (có thể là câu hỏi viết tắt, câu hỏi tiếp nối hoặc hỏi làm rõ),\n"
            "hãy viết lại câu hỏi đó thành một câu hỏi độc lập (Standalone Query), tự chứa đầy đủ thông tin về phương tiện, hành vi và tình tiết đã được đề cập trước đó.\n"
            "Lưu ý:\n"
            "- Nếu câu hỏi mới đã rõ ràng, đầy đủ thông tin hoặc là câu hỏi mở đầu mới hoàn toàn, hãy GIỮ NGUYÊN câu hỏi gốc.\n"
            "- Tuyệt đối KHÔNG tự trả lời câu hỏi, CHỈ viết lại câu hỏi dưới dạng câu hỏi độc lập đầy đủ ngữ cảnh bằng tiếng Việt."
        )
        self.contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # 2. Prompt để Sinh câu trả lời tự nhiên chuẩn pháp lý (Response Generator Prompt)
        generator_system_prompt = (
            "Bạn là \"Antigravity Traffic Assistant\" - Hệ thống chuyên gia tư vấn Luật Giao thông Đường bộ Việt Nam, "
            "tư vấn dựa trên Nghị định 168/2024/NĐ-CP.\n"
            "Nhiệm vụ của bạn là giải thích kết quả xử lý từ Hệ chuyên gia pháp lý thành một phản hồi cực kỳ NGẮN GỌN, SÚC TÍCH, đi thẳng vào vấn đề nhưng chuẩn xác tuyệt đối về pháp lý.\n\n"
            "Dữ liệu từ Hệ Chuyên Gia (Nguồn dữ liệu tối thượng - BẮT BUỘC tuân thủ, KHÔNG tự ý thay đổi số liệu/điều khoản):\n"
            "{expert_system_json}\n\n"
            "Yêu cầu trình bày (Viết cực kỳ cô đọng, loại bỏ hoàn toàn các câu từ xã giao rườm rà):\n"
            "1. **Mức xử phạt**: Ghi rõ tổng số tiền phạt (hoặc khoảng phạt), số điểm GPLX bị trừ, thời gian tước bằng (nếu có).\n"
            "2. **Căn cứ pháp lý**: Liệt kê siêu ngắn từng lỗi kèm theo Điều, Khoản, Điểm và mức phạt thực tế (nêu ngắn gọn tác động tăng nặng/giảm nhẹ nếu có).\n"
            "3. **Các kịch bản cồn (Chỉ ghi nếu dữ liệu có 'alcohol_scenarios_for_vehicle')**: Liệt kê 3 khung cồn dạng gạch đầu dòng cực ngắn kèm tổng mức phạt sau cộng dồn để người dùng tự đối chiếu.\n"
            "4. **Lưu ý**: Đúng một câu khuyên an toàn ngắn gọn dưới 15 từ.\n"
            "5. Nếu là trạng thái thiếu thông tin ('AMBIGUOUS_FALLBACK'), đặt câu hỏi làm rõ siêu ngắn gọn.\n"
            "6. Tuyệt đối KHÔNG tự bịa đặt bất kỳ thông tin nào ngoài dữ liệu JSON."
        )
        self.generator_prompt = ChatPromptTemplate.from_messages([
            ("system", generator_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Tạo Runnable Chains
        self.contextualize_chain = self.contextualize_prompt | self.llm
        self.generator_chain = self.generator_prompt | self.llm

    def _trim_history(self):
        """Giới hạn lịch sử lưu trữ tối đa 5 lượt gần nhất (10 tin nhắn gồm 5 Human và 5 AI)"""
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

    def _extract_text(self, content) -> str:
        """Trích xuất chuỗi văn bản từ nội dung phản hồi của LLM (hỗ trợ cả dạng string và list)"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
            return "".join(texts)
        return str(content)

    def contextualize_query(self, user_query: str) -> str:
        """Sử dụng LLM để biến đổi câu hỏi tiếp nối thành câu hỏi độc lập (Standalone Query)"""
        if not self.chat_history:
            return user_query
        
        response = self.contextualize_chain.invoke({
            "chat_history": self.chat_history,
            "input": user_query
        })
        
        standalone_query = self._extract_text(response.content).strip()
        return standalone_query

    def generate_response(self, original_query: str, standalone_query: str, decision_json: dict) -> str:
        """Sinh câu trả lời hội thoại tự nhiên từ JSON hệ chuyên gia và cập nhật bộ nhớ"""
        response = self.generator_chain.invoke({
            "chat_history": self.chat_history,
            "expert_system_json": json.dumps(decision_json, ensure_ascii=False, indent=2),
            "input": original_query
        })
        
        answer = self._extract_text(response.content).strip()
        
        # Cập nhật lịch sử hội thoại với lượt chat mới nhất
        self.chat_history.append(HumanMessage(content=original_query))
        self.chat_history.append(AIMessage(content=answer))
        self._trim_history()
        
        return answer

    def _write_log_entry(self, log_entry: dict, filename: str = "execution_evidence_logs.json"):
        """Ghi vết nhật ký thực thi chi tiết (Step 3 + Step 4 + Step 5) ra file JSON để làm minh chứng"""
        logs = []
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        
        logs.append(log_entry)
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể lưu log minh chứng: {e}")

    def process_chat_turn(self, user_query: str) -> dict:
        """Xử lý một lượt hội thoại hoàn chỉnh từ câu hỏi thô của người dùng"""
        from datetime import datetime

        # 1. Ngữ cảnh hóa câu hỏi dựa vào lịch sử
        standalone_query = self.contextualize_query(user_query)
        
        # 2. Gọi Step 3 để trích xuất tri thức sự kiện từ standalone query
        event: SuKienGiaoThong = self.pipeline.extractor.extract_event_from_query(standalone_query)
        
        # 3. Định tuyến và gọi Step 4 (Reasoning SQL)
        # Nếu thiếu loại phương tiện hoặc không có bất cứ hành vi vi phạm nào được bóc tách
        if event.loai_phuong_tien == "ChuaXacDinh" or not event.danh_sach_hanh_vi:
            decision = {
                "mode": "AMBIGUOUS_FALLBACK",
                "is_ambiguous": True,
                "thong_tin_con_thieu": event.thong_tin_con_thieu or ["Loại phương tiện di chuyển"],
                "danh_sach_hanh_vi": [hv.mo_ta for hv in event.danh_sach_hanh_vi]
            }
        else:
            decision = self.pipeline.reasoning_engine.evaluate(event)
            
        # 4. Sinh câu trả lời tự nhiên
        answer = self.generate_response(user_query, standalone_query, decision)
        
        # 5. Ghi vết log minh chứng (Step 3, Step 4, Step 5)
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query_original": user_query,
            "standalone_query": standalone_query,
            "step3_event_extracted": event.model_dump(),
            "step4_decision_and_reasoning": decision,
            "step5_conversational_response": answer
        }
        self._write_log_entry(log_entry)
        
        return {
            "standalone_query": standalone_query,
            "event_extracted": event.model_dump(),
            "decision": decision,
            "answer": answer
        }

    def clear_memory(self):
        """Xóa sạch bộ nhớ hội thoại"""
        self.chat_history.clear()


# ==========================================
# KHỞI CHẠY GIAO DIỆN DEMO TRÊN TERMINAL (CLI)
# ==========================================
if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("LỖI: Chưa cấu hình GEMINI_API_KEY trong file .env!")
    else:
        print("\n" + "="*80)
        print("  HỆ THỐNG CHUYÊN GIA TƯ VẤN LUẬT GIAO THÔNG ĐƯỜNG BỘ (Nghị định 168/2024/NĐ-CP)")
        print("                 --- PHIÊN BẢN CHATBOT HỘI THOẠI ĐA LƯỢT ---")
        print("   (Giữ bộ nhớ 5 lượt gần nhất | Gõ 'clear' để xóa bộ nhớ | Gõ 'exit' để thoát)")
        print("="*80 + "\n")
        
        agent = ConversationalTrafficAgent(
            gemini_api_key=GEMINI_API_KEY,
            db_config=POSTGRES_CONFIG
        )
        
        while True:
            try:
                user_input = input("👤 Bạn: ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit", "thoát"]:
                    print("\n🤖 Cảm ơn bạn đã sử dụng hệ thống. Chúc bạn lái xe an toàn!")
                    break
                    
                if user_input.lower() in ["clear", "reset", "xóa bộ nhớ"]:
                    agent.clear_memory()
                    print("🤖 Hệ thống: Đã xóa sạch lịch sử hội thoại!")
                    print("-"*50)
                    continue
                
                print("🤖 Đang suy luận và lập luận...")
                result = agent.process_chat_turn(user_input)
                
                print(f"\n[DEBUG] Câu hỏi ngữ cảnh hóa (Standalone Query): '{result['standalone_query']}'")
                print(f"[DEBUG] Phương tiện bóc tách: {result['event_extracted']['loai_phuong_tien']}")
                print("-" * 50)
                print(f"🤖 Antigravity Assistant:\n\n{result['answer']}")
                print("=" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\n🤖 Tạm biệt!")
                break
            except Exception as e:
                print(f"\n❌ Đã xảy ra lỗi: {e}\n")
