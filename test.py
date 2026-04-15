import math

def calculate_shoelace_area(coords):
    """
    Tính diện tích đa giác 4 cạnh bằng công thức Shoelace.
    Input: Danh sách 8 số nguyên [x0, y0, x1, y1, x2, y2, x3, y3]
    """
    if len(coords) != 8:
        raise ValueError(f"Dữ liệu lỗi! Cần đúng 8 số (4 tọa độ), nhưng bạn đang nhập {len(coords)} số.")
    
    # Tách 8 số thành 4 cặp tọa độ (x, y)
    x0, y0 = coords[0], coords[1]
    x1, y1 = coords[2], coords[3]
    x2, y2 = coords[4], coords[5]
    x3, y3 = coords[6], coords[7]
    
    # Áp dụng công thức Shoelace
    area = 0.5 * abs((x0*y1 - x1*y0) + (x1*y2 - x2*y1) + (x2*y3 - x3*y2) + (x3*y0 - x0*y3))
    
    return area

def main():
    print("--- CHƯƠNG TRÌNH TÍNH DIỆN TÍCH QR CODE ---")
    
    # Bạn có thể thay đổi chuỗi string này thành dữ liệu của bạn
    # Ví dụ dưới đây là một tọa độ hình vuông chuẩn 100x100
    raw_input = "95,71,148,71,148,116,95,116" 
    
    print(f"Chuỗi đầu vào: {raw_input}")
    
    try:
        # Xử lý chuỗi: cắt theo dấu phẩy, xóa khoảng trắng và ép kiểu sang int
        coords = [int(num.strip()) for num in raw_input.split(',')]
        
        # Gọi hàm tính diện tích
        area = calculate_shoelace_area(coords)
        
        print("\n[THÀNH CÔNG]")
        print(f"- Tọa độ Point 0: ({coords[0]}, {coords[1]})")
        print(f"- Tọa độ Point 1: ({coords[2]}, {coords[3]})")
        print(f"- Tọa độ Point 2: ({coords[4]}, {coords[5]})")
        print(f"- Tọa độ Point 3: ({coords[6]}, {coords[7]})")
        print(f"=> Diện tích mã QR: {area} pixel vuông")
        
    except ValueError as e:
        print(f"\n[LỖI] {e}")
        print("Vui lòng kiểm tra lại chuỗi tọa độ (đảm bảo chỉ có 8 con số cách nhau bằng dấu phẩy).")
    except Exception as e:
        print(f"\n[LỖI KHÔNG XÁC ĐỊNH] {e}")

if __name__ == "__main__":
    main()