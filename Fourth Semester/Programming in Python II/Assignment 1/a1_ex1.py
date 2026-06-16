class Product:
    def __init__(self, name: str, category: str, price: float, rating: float,
                 launch_date: str, in_stock: bool):
        self.name = name
        self.category = category
        self.price = price
        self.rating = rating
        self.launch_date = launch_date
        self.in_stock = in_stock

    def __str__(self) -> str:
        return f"{self.name} ({self.category}, {self.rating}, {self.launch_date}) - {self.price:.2f}"


def organize_catalog(products: list[Product], mode: str = "cheapest") -> list[Product]:
    if mode == "all":
        # Show all products, sorted by name ascending --> no distinction with in stock products
        return sorted(products, key=lambda p: p.name)
    else:
        # Only products in stock
        in_stock_products = [p for p in products if p.in_stock]

        if mode == "cheapest":
            return sorted(in_stock_products, key=lambda p: p.price)
        elif mode == "category_then_price":
            return sorted(in_stock_products, key=lambda p: (p.category, p.price))
        elif mode == "best_rated":
            return sorted(in_stock_products, key=lambda p: p.rating, reverse=True)
        elif mode == "newest":
            # From example, the date is in format YYYY-MM-DD, reverse sort will give newest first
            return sorted(in_stock_products, key=lambda p: p.launch_date, reverse=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Copy-Paste of the exercise
    catalog = [
        Product(name="Fancy Headphones", category="Electronics",
                price=199.99, rating=4.7, launch_date="2022-11-05", in_stock=True),
        Product(name="Wireless Mouse", category="Electronics",
                price=29.99, rating=4.3, launch_date="2023-03-01", in_stock=True),
        Product(name="Hiking Backpack", category="Outdoors",
                price=59.50, rating=4.8, launch_date="2021-07-15", in_stock=False),
        Product(name="Novel: The Great Adventure", category="Books",
                price=15.99, rating=4.9, launch_date="2023-01-10", in_stock=True),
        Product(name="Game Console", category="Electronics",
                price=299.0, rating=4.5, launch_date="2022-09-20", in_stock=True),
    ]

    print("\n--- cheapest (in_stock only) ---")
    for p in organize_catalog(catalog, "cheapest"):
        print(p)

    print("\n--- best_rated (in_stock only) ---")
    for p in organize_catalog(catalog, "best_rated"):
        print(p)

    print("\n--- newest (in_stock only) ---")
    for p in organize_catalog(catalog, "newest"):
        print(p)

    print("\n--- category_then_price (in_stock only) ---")
    for p in organize_catalog(catalog, "category_then_price"):
        print(p)

    print("\n--- all (including out_of_stock), default sort by name ---")
    for p in organize_catalog(catalog, "all"):
        print(p)

    print("\n--- wrong mode (lowest_rated) ---")
    try:
        organize_catalog(catalog, "lowest_rated")
    except ValueError as e:
        print(e)
