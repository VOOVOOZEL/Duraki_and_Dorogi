import socket
import threading
import os
import select
import json

import sys
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')

DB_IP = "192.168.71.108"
DB_PORT = 53192

class ThreadedServer(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.result = {}

    def listen(self):
        # change this property
        NOMEROFF_NET_DIR = os.path.abspath('./')

        # specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
        MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')

        MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')
        MASK_RCNN_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/mask_rcnn_numberplate_0700.h5")
        OPTIONS_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/numberplate_options_2019_05_15.h5")

        # If you use gpu version tensorflow please change model to gpu version named like *-gpu.pb
        mode = "cpu"
        OCR_NP_UKR_TEXT = os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_ua_12-{}.h5".format(mode))
        OCR_NP_EU_TEXT = os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_eu_2-{}.h5".format(mode))
        OCR_NP_RU_TEXT = os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_ru_3-{}.h5".format(mode))

        sys.path.append(NOMEROFF_NET_DIR)

        # Import license plate recognition tools.
        from NomeroffNet import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing

        # Initialize npdetector with default configuration file.
        nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)

        # Load weights in keras format.
        nnet.loadModel(MASK_RCNN_MODEL_PATH)

        # Initialize rect detector with default configuration file.
        rectDetector = RectDetector()

        # Initialize text detector.
        # Also you may use gpu version models.
        textDetector = TextDetector({
            "eu_ua_2004_2015": {
                "for_regions": ["eu_ua_2015", "eu_ua_2004"],
                "model_path": OCR_NP_UKR_TEXT
            },
            "eu": {
                "for_regions": ["eu", "eu_ua_1995"],
                "model_path": OCR_NP_EU_TEXT
            },
            "ru": {
                "for_regions": ["ru"],
                "model_path": OCR_NP_RU_TEXT
            }
        })

        # Initialize train detector.
        optionsDetector = OptionsDetector()
        optionsDetector.load(OPTIONS_MODEL_PATH)

        self.sock.listen(5)
        while True:
            client, address = self.sock.accept()
            print("Received connection:", address)
            client.settimeout(30)
            tmp = threading.Thread(target=self.listenToClient, args=(client, address, self.result))
            tmp.start()
            tmp.join()

            print(self.result)

            if not self.result.values(): continue
            # Detect numberplate
            img_path = self.result[client][0]
            print("Saved tmp image file", img_path)
            img = mpimg.imread(img_path)
            try:
                NP = nnet.detect([img])
            except IndexError:
                continue

            # Generate image mask.
            cv_img_masks = filters.cv_img_mask(NP)

            # Detect points.
            arrPoints = rectDetector.detect(cv_img_masks)
            zones = rectDetector.get_cv_zonesBGR(img, arrPoints)

            # find standart
            # Added a classifier (isHiddenIds) for determining the fact of hide text of number, in order not to recognize a deliberately damaged license plate image.
            regionIds, isHiddenIds = optionsDetector.predict(zones)
            regionNames = optionsDetector.getRegionLabels(regionIds)

            # find text with postprocessing by standart
            textArr = textDetector.predict(zones, regionNames)
            textArr = textPostprocessing(textArr, regionNames)
            print("Parsed array of signs:",textArr)
            dbSocket = socket.socket()
            dbSocket.connect((DB_IP, DB_PORT))
            for num in textArr:
                if num == "":
                    continue
                car = {num: self.result[client][1]}
                app_json = json.dumps(car)
                dbSocket.send(app_json.encode())
                ready = select.select([dbSocket], [], [], 10)
                if ready[0]:
                    recv = dbSocket.recv(4096)
                else:
                    continue
                print("Rating of", num, " is ", json.loads(recv.decode())["rating"])
                rate = json.loads(recv.decode())["rating"]
                alarm = "False"
                if rate < 5:
                    alarm = "True"
                response = {"number": num,"alarm": alarm}
                print("Sended respone", (json.dumps(response)))
                client.send((json.dumps(response)).encode())
            #dbSocket.close()
            os.remove(img_path)
            self.result.pop(client)


    def listenToClient(self, client, address, result):
        reply = b''
        path = str(address[0]) + "_" + str(address[1]) + ".jpg"
        while select.select([client], [], [], 3)[0]:
            data = client.recv(2048)
            if not data: break
            reply += data
        headers = reply.split(b'\r\n\r\n')[0]
        method = str(reply[:4])
        print("Request method:", method)
        if "GET" in method:
            image = reply[len(headers) + 4:]

            # save image
            f = open(path, 'wb')
            f.write(image)
            f.close()
            self.result[client] = path, method, headers
        elif "POST" in method:
            req = json.loads(reply[len(headers) + 4:].decode())
            self.result[client] = req["member"]

if __name__ == "__main__":
    while True:
        port_num = input("Port? ")
        try:
            port_num = int(port_num)
            break
        except ValueError:
            pass

    ThreadedServer('', port_num).listen()
