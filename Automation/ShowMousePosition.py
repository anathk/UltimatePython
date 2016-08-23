import pyautogui

x, y = (0, 0)
try:
    while True:
        newX, newY = pyautogui.position()
        if (newX, newY) != (x, y):
            positionStr = 'X: ' + str(newX).rjust(4) + ' Y: ' + str(newY).rjust(4)
            x, y = newX, newY
            print(positionStr)

except KeyboardInterrupt:
    print("\nDnoe.")