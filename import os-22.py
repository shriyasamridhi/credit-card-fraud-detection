import cv2
import numpy as np

# Show warning/info before verification
instructions = """\
IMPORTANT: Do a quick manual check first!

Texture – A fake credit card may feel flimsy or rough.
Run your fingers along the edges to feel for inconsistencies.

Rigidity – Try bending the card slightly to check its stiffness.
A real card will be rigid and resistant to bending.

Press any key to continue with image verification...
"""

# Create a blank window and show the message
msg_img = np.ones((300, 600, 3), dtype=np.uint8) * 255  # White background
y0, dy = 30, 30
for i, line in enumerate(instructions.split("\n")):
    y = y0 + i * dy
    cv2.putText(msg_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

cv2.imshow("Manual Card Check Instructions", msg_img)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()

# --- Continue with image processing ---
# Load images
card_image = cv2.imread('D:\\ns_pbl\\sbi_soumya.jpg')
logo_image = cv2.imread('sbi-logo.png')

# Step 1: Check if images are loaded properly
if card_image is None or logo_image is None:
    print("Error: Image not loaded properly.")
    exit()
else:
    print("Images loaded successfully.")

# Step 2: Check dimensions of images
print("Card Image Size: ", card_image.shape)
print("Logo Image Size: ", logo_image.shape)

# Step 3: Convert to grayscale
gray_card = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
gray_logo = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)

print("Gray Card Image Size: ", gray_card.shape)
print("Gray Logo Image Size: ", gray_logo.shape)

# Step 4: Ensure the logo is smaller than the card image
if gray_logo.shape[0] > gray_card.shape[0] or gray_logo.shape[1] > gray_card.shape[1]:
    print("Logo is larger than card. Resizing logo...")
    new_size = (min(gray_logo.shape[1], gray_card.shape[1]), min(gray_logo.shape[0], gray_card.shape[0]))
    gray_logo = cv2.resize(gray_logo, new_size, interpolation=cv2.INTER_AREA)
    print("Logo resized to: ", gray_logo.shape)
else:
    print("Logo is smaller than or equal to card. No resizing needed.")

# Step 5: Perform template matching
result = cv2.matchTemplate(gray_card, gray_logo, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print("Result Matrix: \n", result)
print("Max Value: ", max_val)
print("Max Location: ", max_loc)

# Step 6: Threshold check for match
threshold = 0.6
if max_val >= threshold:
    print("Card logo is legitimate.")
else:
    print("Card logo is not detected.")

