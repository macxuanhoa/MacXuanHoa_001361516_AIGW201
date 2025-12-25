from tools import predict_house_price

result = predict_house_price(
    area=2000,
    bedrooms=3,
    bathrooms=2,
    stories=2,
    parking=1,
    neighborhood="A",
    house_style="Villa"
)

print(result)
