import qrcode

data = "gopi"

img = qrcode.make(data)

img.save("my_qrcode.png")

print("QR Code was generated and saved as my_qrcode.png!")

