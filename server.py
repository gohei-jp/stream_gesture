import asyncio
import base64
import cv2
import numpy as np
import websockets
from modules.gesture import GestureApp

gesture_app = GestureApp()

async def process_image(websocket, path):
    while True:
        data = await websocket.recv()
        img_data = base64.b64decode(data[23:])
        with open("test_image.jpg", "wb") as f:
            f.write(img_data)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        try:
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except cv2.error as e:
            print(f"Error decoding image: {e}")
            continue

        processed_img = gesture_app.update(img)

        # processed_img = process_frame(img) # ここで画像処理を行う
        
        _, buffer = cv2.imencode('.jpeg', processed_img)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        await websocket.send(f'data:image/jpeg;base64,{encoded_img}')

def process_frame(frame):
    # 画像をグレースケールに変換
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

if __name__ == '__main__':
    server = websockets.serve(process_image, 'localhost', 8000)
    asyncio.get_event_loop().run_until_complete(server)
    print("WebSocket server is running on ws://localhost:8000")
    asyncio.get_event_loop().run_forever()