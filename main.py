import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
from skimage.measure import label,regionprops
import requests
import imutils

# klasa detektora zawierająca metody do ekstrakcji cech i detekcji okiektów
class bombki:
    def __init__(self):
        self.model = None
        self.lista_cech =  ['EulerNumber','Area','BoundingBoxArea','FilledArea','Extent','EquivDiameter','Solidity', 'eccentricity', 'Area_FilledArea', 'major2minor', 'major_axis_length', 'minor_axis_length', 'bbarea_area']
        self.cechy = pd.DataFrame(columns=self.lista_cech + ["Hu1_log", "Hu2_log", "Hu3_log", "Hu4_log", "Hu5_log", "Hu6_log", "Hu7_log"])
        self.klasa = {
            0: 'konik',
            1: 'ryba',
            2: 'butelka'
        }

    # funkcja znajduje obiekty na obrazie, zwraca ich cechy i bounding boxy
    def ekstrakcja_cech(self, o: np.array):
        # przekształcenie obrazu w przestrzeń barw hsv
        hsv = cv2.cvtColor(o, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

        # maska od czerownego po zielony
        lower_red2green = np.array([0, 50, 55])
        upper_red2green = np.array([90, 255, 255])
        mask0 = cv2.inRange(hsv, lower_red2green, upper_red2green)

        # maska czerwony
        lower_red = np.array([160, 50, 55])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # maska beżowy i biały
        lower_white = np.array([0, 0, 155])
        upper_white = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_white, upper_white)

        # maska ciemny zielony
        lower_dark_greem = np.array([50, 170, 13])
        upper_dark_green = np.array([70, 255, 120])
        mask3 = cv2.inRange(hsv, lower_dark_greem, upper_dark_green)

        # powiększenie obszaru filtra ciemny zielony
        kernel = np.ones((3, 3), np.uint8)
        mask3 = cv2.dilate(mask3, kernel, iterations=3)

        # stworzenie maski o wszystkich kolorach bąbek i ewentualnych cieniach/jasnych odbiciach
        mask = mask0 + mask1 + mask2 + mask3

        # nakładanie maski na zdjęcie
        output_hsv = o.copy()
        output_hsv[np.where(mask == 0)] = 0

        # zmiana przestrzeni barw na odcienie szarości
        gray = cv2.cvtColor(output_hsv, cv2.COLOR_BGR2GRAY)
        # binaryzacja zdjęcia
        ret, b = cv2.threshold(gray, 20, 255, 0)
        # operacja otwierania
        elstr = np.ones((5, 5), np.uint8)
        close = cv2.morphologyEx(b, cv2.MORPH_CLOSE, elstr, iterations=3)

        # etykietowanie i ekstrakcja cech
        cechy_ = regionprops(label(close))

        lista_bounds = []
        ile_obiektow = len(cechy_)

        # przejście po wszystkich wykrytych obiektach
        for i in range(0, ile_obiektow):
            cechy_dict = {}
            # jeśli obiekt jest za mały na wyliczenie cech potrzebnych w klasyfikacji, jest odrzucany
            if cechy_[i][self.lista_cech[1]] < 6000:
                continue
            # wyznaczenie bounding box ów
            yp, xp, yk, xk = cechy_[i]['BoundingBox']
            lista_bounds.append((yp, xp, yk, xk))
            aktualny_obiekt = o[yp:yk, xp:xk, :]
            ret, binobj = cv2.threshold(aktualny_obiekt[:, :, 1], 0, 255, cv2.THRESH_BINARY)

            # odczytanie i obliczenie cech z lista_cech
            cechy_dict[self.lista_cech[0]] = cechy_[i][self.lista_cech[0]]
            cechy_dict[self.lista_cech[1]] = cechy_[i][self.lista_cech[1]]
            cechy_dict[self.lista_cech[2]] = cechy_[i][self.lista_cech[2]]
            cechy_dict[self.lista_cech[3]] = cechy_[i][self.lista_cech[3]]
            cechy_dict[self.lista_cech[4]] = cechy_[i][self.lista_cech[4]]
            cechy_dict[self.lista_cech[5]] = cechy_[i][self.lista_cech[5]]
            cechy_dict[self.lista_cech[6]] = cechy_[i][self.lista_cech[6]]
            cechy_dict[self.lista_cech[7]] = cechy_[i][self.lista_cech[7]]
            cechy_dict[self.lista_cech[8]] = cechy_[i][self.lista_cech[1]] / cechy_[i][self.lista_cech[3]]
            cechy_dict[self.lista_cech[9]] = cechy_[i][self.lista_cech[10]] / cechy_[i][self.lista_cech[11]]
            cechy_dict[self.lista_cech[10]] = cechy_[i][self.lista_cech[10]]
            cechy_dict[self.lista_cech[11]] = cechy_[i][self.lista_cech[11]]
            cechy_dict[self.lista_cech[12]] = cechy_[i][self.lista_cech[2]] / cechy_[i][self.lista_cech[1]]

            # rejestrujemy wybrane cechy wyznaczone przez regionprops
            # dodajemy momenty Hu
            hu = cv2.HuMoments(cv2.moments(binobj))
            hulog = (1 - 2 * (hu > 0).astype('int')) * np.nan_to_num(np.log10(np.abs(hu)), copy=True, neginf=-99,
                                                                     posinf=99)
            hulog_flat = hulog.flatten()

            cechy_dict["Hu1_log"] = hulog_flat[0]
            cechy_dict["Hu2_log"] = hulog_flat[1]
            cechy_dict["Hu3_log"] = hulog_flat[2]
            cechy_dict["Hu4_log"] = hulog_flat[3]
            cechy_dict["Hu5_log"] = hulog_flat[4]
            cechy_dict["Hu6_log"] = hulog_flat[5]
            cechy_dict["Hu7_log"] = hulog_flat[6]

            cechy_dict["ktory_obiekt"] = i

            print(cechy_dict)

            self.cechy.loc[len(self.cechy)] = cechy_dict
        return close, lista_bounds

        #tabela_cech[:, 0] = (tabela_cech[:, 0] == 1)  # korekta liczby Eulera

    # funkcja przeprowadza predykcję, dla podanego zdjęcia, przekazywane są wybrane cechy, które zależą od cech użytych do uczenia modelu
    def predict(self, image: np.array, wybrane_cechy: list):
        _, lista_bounds = self.ekstrakcja_cech(image)
        x_test = self.cechy[wybrane_cechy]
        # poniżej rysowanie bounding boxów i dodawanie podpisu na przekazanym do funkcji zdjęciu - wykonane dla każdego wykrytego obiektu
        for i in range(len(lista_bounds)):
            print(len(lista_bounds))
            bounds = lista_bounds[-1-i]
            pred = self.model.predict([x_test.iloc[len(x_test)-i-1]])
            print(pred)
            image = self.draw_boundingbox(image, bounds, str(self.klasa[pred[0]]))
        return image

    # załadowanie nauczonego modelu knn
    def load_model(self, filename='finalized_model.sav'):
        self.model = pickle.load(open(filename, 'rb'))

    # funkcja, która rysuje bboxy i dodaje podpi
    def draw_boundingbox(self, image, bounds, pred):
        yp, xp, yk, xk = bounds
        cv2.putText(img, pred, (xp,yp), cv2.FONT_HERSHEY_SIMPLEX, 1, (168, 50, 123), 5)
        return cv2.rectangle(image, (xp, yp), (xk, yk), (0, 0, 255), 2)


if __name__ == '__main__':

    # tworzenie obiektu do detekcji
    Bombki = bombki()
    # ładowanie modelu
    Bombki.load_model(filename='finalized_model_14_14.sav')
    # dostęp do kamery w telefonie
    url = "http://192.168.0.136:8080/shot.jpg"

    while (True):
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)
        cv2.imshow("Android_cam", img)
        # try ponieważ raz na jakiś czas zdarza się uszkodzone zdjęcie
        try:
            pred = Bombki.predict(img, 0, ['EulerNumber', 'Solidity', 'Extent', 'eccentricity', 'Area_FilledArea', 'major2minor', 'bbarea_area'])
            cv2.imshow("detect", pred)
        except Exception as e:
            pass
        # Press Esc key to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

