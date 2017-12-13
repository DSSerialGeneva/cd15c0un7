

class ImgMeta:
    def __init__(self, product_id, category_id, no):
        self.product_id = product_id
        self.category_id = category_id
        self.no = no

    def __str__(self) -> str:
        return str(self.product_id) + '-' + str(self.category_id) + '-' + str(self.no)

    def __array__(self):
        return self.product_id, self.category_id, self.no
