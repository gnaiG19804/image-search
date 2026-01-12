class SpatialVerifier:
    """
    Kiem tra vi tri cua thanh phan UI co hop ly hay khong.
    (Spatial Layer - Lop kiem tra vi tri)
    """
    
    @staticmethod
    def verify(component_type: str, bbox: tuple, img_w: int, img_h: int) -> float:
        """
        Tra ve diem so vi tri (0.0 - 1.0)
        - 1.0: Vi tri hop ly (VD: Header o tren cung)
        - 0.5: Vi tri dang ngo
        - 0.0: Vi tri sai lech hoan toan (VD: Footer o tren dau)
        """
        x, y, w, h = bbox
        cy = y + h/2
        cx = x + w/2
        
        normalized_y = cy / img_h
        normalized_x = cx / img_w
        
        ctype = component_type.lower().replace(" ", "_s_") # chuan hoa tam thoi
        
        # 1. HEADER / NAVBAR
        if 'header' in ctype or 'nav' in ctype:
            # Phai nam o top 45% (Relaxed from 30% to handle top banners)
            if normalized_y < 0.45:
                return 1.0
            elif normalized_y < 0.6:
                return 0.5 # Chap nhan duoc
            else:
                return 0.1 # Header khong the o duoi chan trang
                
        # 2. FOOTER
        if 'footer' in ctype:
            # Phai nam o bottom 30%
            if normalized_y > 0.7:
                return 1.0
            elif normalized_y > 0.5:
                return 0.5
            else:
                return 0.1
                
        # 3. SIDEBAR
        if 'sidebar' in ctype:
            # Chieu cao phai lon (tam > 40% man hinh)
            if h / img_h < 0.4:
                return 0.3 # Qua ngan de la sidebar
                
            # Phai nam sat le trai hoac le phai
            if normalized_x < 0.25 or normalized_x > 0.75:
                return 1.0
            else:
                return 0.2 # Sidebar khong nam giua man hinh
        
        # 4. HERO SECTION
        if 'hero' in ctype:
             # Thuong o nua tren man hinh
             if normalized_y < 0.6:
                 return 1.0
             else:
                 return 0.4
                 
        # Mac dinh (Login form, card, button...) -> Khong rang buoc vi tri
        return 0.9
