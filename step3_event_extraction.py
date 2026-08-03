import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class HanhViDetail(BaseModel):
    ma_hanh_vi: str = Field(
        description="Mã hành vi vi phạm chuẩn hóa. BẮT BUỘC chọn từ danh sách: "
                    "['NONG_DO_CON', 'TOC_DO', 'QUEN_GPLX', 'KHONG_GPLX', "
                    "'DANG_KIEM_HET_HAN_DUEI_1T', 'DANG_KIEM_HET_HAN_TU_1T', "
                    "'KHONG_CHAP_HANH_CSGT', 'VUOT_DEN_DO', 'KHONG_DONG_MU_BH', "
                    "'KHONG_THAT_DAY_AN_TOAN', 'DUNG_DIEN_THOAI', "
                    "'GIAO_XE_CHO_NGUOI_KHONG_DU_DIEU_KIEN', 'GAY_TAI_NAN_BO_CHAY']" # <--- Thêm mã mới
    )
    chi_so_dinh_luong: Optional[float] = Field(
        default=None, 
        description="Chỉ số định lượng cụ thể: Nồng độ cồn hơi thở (mg/L), Vận tốc chạy quá (km/h) nếu có."
    )
    mo_ta: str = Field(description="Mô tả ngắn gọn hành vi vi phạm bằng tiếng Việt.")


class SuKienGiaoThong(BaseModel):
    loai_phuong_tien: str = Field(
        description="Loại phương tiện: 'XeMay', 'OTo', 'XeDap', 'XeMayChuyenDung', 'NguoiBoHanh', hoặc 'ChuaXacDinh'"
    )
    doi_tuong_bi_xu_phat: str = Field(
        default="NguoiDieuKhien",
        description="Đối tượng bị xử phạt: 'NguoiDieuKhien' (Người lái xe) hoặc 'ChuPhuongTien' (Chủ xe). Mặc định là 'NguoiDieuKhien'."
    )
    danh_sach_hanh_vi: List[HanhViDetail] = Field(
        default_factory=list, 
        description="Danh sách các chi tiết hành vi vi phạm."
    )
    tinh_tiet_tang_nang: List[str] = Field(
        default_factory=list, 
        description="Tình tiết tăng nặng (VD: 'chống đối', 'không chấp hành hiệu lệnh dừng xe', 'cãi lộn')."
    )
    tinh_tiet_giam_nhe: List[str] = Field(
        default_factory=list, 
        description="Tình tiết giảm nhẹ."
    )
    is_ambiguous: bool = Field(
        description="Chỉ bằng TRUE khi KHÔNG BIẾT LOẠI XE ('ChuaXacDinh'). Nếu ĐÃ BIẾT LOẠI XE thì gán FALSE."
    )
    thong_tin_con_thieu: List[str] = Field(
        default_factory=list,
        description="Danh sách mô tả các thông tin còn thiếu."
    )


class EventExtractor:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.1-flash-lite"

    def extract_event_from_query(self, user_query: str) -> SuKienGiaoThong:
        system_instruction = """
        Bạn là chuyên gia trích xuất tri thức pháp lý cho Luật Giao thông Đường bộ Việt Nam (Nghị định 168/2024/NĐ-CP).
        Nhiệm vụ: Phân tích mô tả của người dùng và trích xuất thông tin vào Schema JSON.

        Quy tắc gán 'doi_tuong_bi_xu_phat':
        - Nếu hỏi về việc "cho mượn xe", "giao xe cho người chưa đủ tuổi/không có bằng" -> gán 'ChuPhuongTien'.
        - Mọi trường hợp trực tiếp điều khiển xe -> gán 'NguoiDieuKhien'.

        Quy tắc gán 'ma_hanh_vi':
        - Uống rượu/bia, đo cồn -> 'NONG_DO_CON'
        - Chạy quá tốc độ -> 'TOC_DO'
        - Quên mang/không mang GPLX -> 'QUEN_GPLX'
        - Không có GPLX / chưa đủ tuổi lái xe -> 'KHONG_GPLX'
        - Đăng kiểm hết hạn dưới 1 tháng -> 'DANG_KIEM_HET_HAN_DUEI_1T'
        - Đăng kiểm hết hạn từ 1 tháng trở lên -> 'DANG_KIEM_HET_HAN_TU_1T'
        - Vượt đèn đỏ -> 'VUOT_DEN_DO'
        - Không đội mũ bảo hiểm -> 'KHONG_DONG_MU_BH'
        - Không thắt dây an toàn -> 'KHONG_THAT_DAY_AN_TOAN'
        - Dùng điện thoại khi lái xe -> 'DUNG_DIEN_THOAI'
        - Gây tai nạn giao thông rồi bỏ chạy, không dừng lại, không giữ nguyên hiện trường -> 'GAY_TAI_NAN_BO_CHAY'
        - Không chấp hành hiệu lệnh CSGT (Lưu ý: bỏ chạy do trốn CSGT thì dùng mã này, bỏ chạy do gây tai nạn dùng mã GAY_TAI_NAN_BO_CHAY) -> 'KHONG_CHAP_HANH_CSGT'
        """

        prompt = f"Mô tả của người dùng:\n\"{user_query}\""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=SuKienGiaoThong,
                temperature=0.0
            )
        )

        return SuKienGiaoThong.model_validate_json(response.text)