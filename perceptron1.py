#Matematikai muveletek elvegzesehez
import numpy as np
# koordinata rendszer kirajzolasahoz
import matplotlib.pyplot as plt
# 3D kirajzolashoz
from mpl_toolkits.mplot3d import axes3d 
# Matlab plot
from matplotlib import cm


def Cross_Entropy(y_hat, y):
    # 2 lehetseges esett:  0 or 1
    # np.log() termeszetes logaritmust jelent
    if y == 1:
      return -np.log(y_hat)
    else:
      return -np.log(1 - y_hat)

# Klasszikus szigmoid fuggveny
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_Cross_Entropy(y_hat, y):
    # Y-nak ket esette van ujra y=0/1
    if y == 1:
      return -1/y_hat
    else:
      return 1 / (1 - y_hat)

# sigmoid derivaltja
def derivative_sigmoid(x):
    return x*(1-x)

# Az adataink
X = np.array([[0, 0], [0, 5], [5, 0], [5, 5]])

# Igazsag Tablazat
Y = np.array([0, 0, 0, 1])

area = 200
fig = plt.figure(figsize=(6, 6))
plt.title('Az ES kapu', fontsize=20)
ax = fig.add_subplot(111)
#szinek osztalya: 0 piros, 1 kek
ax.scatter(0, 0, s=area, c='r', label="Class 0")
ax = fig.add_subplot(111)
ax.scatter(0, 5, s=area, c='r', label="Class 0")
ax = fig.add_subplot(111)
ax.scatter(5, 0, s=area, c='r', label="Class 0")
ax = fig.add_subplot(111)
ax.scatter(5, 5, s=area, c='b', label="Class 1")
plt.grid()


low = -0.01
high = 0.01

W_2 = np.random.uniform(low=low, high=high, size=(1,))
W_1 = np.random.uniform(low=low, high=high, size=(1,))
W_0 = np.random.uniform(low=low, high=high, size=(1,))

#Epoch erteke. Minden ertekre vegig sopor az adatainkon
Epoch = 2000
#A tanulasi rata kozkedveten eleg kicsi kell legyen
eta = 0.01

E = []
# Az aktualis trainig elkezdodik annyiszor amennyi az Epoch
for ep in range(Epoch):
    random_index = np.arange(X.shape[0])
    # random_index ugyanakkora hosszusaggal rendelkezik mint az X es Y, es egy megkevert X index listat tartalmaz. 
    np.random.shuffle(random_index)
    # e menti az hibakat minden epochban. Majd atlagolja es ez hozzaadodik a E-hez.
    # Minden uj ciklusban uresse tesszuk.
    e = []
    # Ez a lista vegig megy a megkevert training datan. a random index biztositja, hogy 
    # az X-ben a legjobb erteket vesszuk ki
    for i in random_index:
        # kiveszi az i-dik erteket X-bol
        x = X[i]
        # Kiszamolja Z, ami a szigmoid mertekeunk lesz
        Z = W_1* x[0] + W_2* x[1] + W_0
        # alkalmazzuk a szigmoid fugvenyre, hogy generaljon egy kimenetet a perceptronnak
        Y_hat = sigmoid(Z)
        # kiszamolja binary cross-entropy errorokat az i-dik elemre es hozzaadja a e[]-hez
        e.append(Cross_Entropy(Y_hat, Y[i]))
        
        # Kiszamolja a hiba fuggvenyunk gradienseit, 3 tanulhato ertek (a neuralis halo sulyai)
        dEdW_1 = derivative_Cross_Entropy(Y_hat, Y[i])*derivative_sigmoid(Y_hat)*x[0]
        dEdW_2 = derivative_Cross_Entropy(Y_hat, Y[i]) * derivative_sigmoid(Y_hat) * x[1]
        dEdW_0 = derivative_Cross_Entropy(Y_hat, Y[i])*derivative_sigmoid(Y_hat)
        
        # Frissitjuk a parametereket a kiszamolt gradiensekkel. Sztochasztikus gradiens!
        W_0 = W_0 - eta * dEdW_0
        W_1 = W_1 - eta*dEdW_1
        W_2 = W_2 - eta* dEdW_2
    
    #Minden 500-dik alkalommal szeretnek latni a valtozast
    if ep % 500 == 0:
        #fiugra generalas
        fig = plt.figure(figsize=(15, 6))
        plt.title('The AND Gate', fontsize=20)
        # al figura beszurasa melyet biztositunk h 3d legyen
        ax = fig.add_subplot(131, projection='3d')
        # ki rajzolunk minden egyes kulon allo pontot
        ax.scatter(0, 0, s=area, c='r', label="Class 0")
        ax.scatter(0, 5, s=area, c='r', label="Class 0")
        ax.scatter(5, 0, s=area, c='r', label="Class 0")
        ax.scatter(5, 5, s=area, c='b', label="Class 1")

        plt.title('Decision Boundary Created by the Hyper-plane')
        # Az egyenlosegben Z = W2X2 + W1X1 + W0 es tudjuk hogy a linearis dontes hatar meghatarzohato hogyha Z = 0,
        # es atrendezve x_2 = (-W1/W2) * x1 - (W0/W2). 
        # Tudva hogy a perceptron mar tudja a sulyokat, x1-nek valasztunk egy terjedelmet (range), 
        # es mindegyikre kiszamoljuk x2-t.  Ezzel kirajzolunk egy folytonos vonalat
        x_1 = np.arange(-2, 5, 0.1)
        W_1 * x[0] + W_2 * x[1] + W_0
        x_2 = (-W_1/W_2) * x_1 - (W_0/W_2)
        plt.grid()
        plt.plot(x_2, x_1, '-k', marker='_', label="DB")
        plt.xlabel('x1', fontsize=20)
        plt.ylabel('x2', fontsize=20)
        # Hozzaadjuk a masodik alfigurat. Ez mutatja az utat amely keresztul vag a bemeneti kozon
        ax = fig.add_subplot(132, projection='3d')
        x_0 = np.arange(-10, 10, 0.1)
        # szuksegunk van a meshgridre hogy harom dimenziosan is tudjunk rajzolni
        X_0, X_1 = np.meshgrid(x_0, x_1)
        # Minden x_0 es X_1 kombinaciojara kiszamoljuk a Z-t harom dimenzioban
        Z = X_0*W_1 + X_1*W_2 + W_0
        # Felhasznaljuk a wireframe libraryt, hogy lathassuk a hyper plane mogott a lepeseket
        # meghatarozzuk a grid meretet.
        ax.plot_wireframe(X_0, X_1, Z, rstride=10, cstride=10)
        # Vizualizalni szeretnenk a linearis dontes hatarokat felhasznalva az elobbi al figurat
        ax.scatter(x_2, x_1, 0, marker='_', c='k')
        # Kirajzoljuk ezeket a pontokat is
        ax.scatter(0, 0, 0, marker='o', c='r', s=100)
        ax.scatter(0, 5, 0, marker='o', c='r', s=100)
        ax.scatter(5, 0, 0, marker='o', c='r', s=100)
        ax.scatter(5, 5, 0, marker='o', c='b', s=100)
        plt.xlabel('x1', fontsize=20)
        plt.ylabel('x2', fontsize=20)
        plt.title('The Hyper-plane Cutting through Input Space')
        plt.grid()
        # A harmadik alFigura mutatja a sigmoid erejet, a kiemelt resz osszenyomodik a 0 es 1 kozott
        ax = fig.add_subplot(133, projection='3d')
        # A cm konyvtar kiszinezi a sigmoid altal kigeneralt alfigurat, a sigmoid ertekei alapjan
        # Vilagosabb szinek kozelebb vannak az egyhez es a sotetek a 0-hoz! 
        # Igy latni fogjuk hogy a szigmoid(Z), ami a vegso kimenet a perceptronnak (y_hat)
        # megkozelitloeg egyenlok lesznek 1-el es az tobbi megkozlitoleg egyenlok lesznek a 0-val.
        my_col = cm.jet(sigmoid(Z) / np.amax(sigmoid(Z)))
        ax.plot_surface(X_0, X_1, sigmoid(Z), facecolors=my_col)
        # Ujra akarjuk latni a linearis dontes hatarokat melyek letrejottek az elso alfiguraban a jelenlegi training peldaban
        ax.scatter(x_2, x_1, 0, marker='_', c='k')
        ax.scatter(0, 0, 0, marker='o', c='r', s=100)
        ax.scatter(0, 5, 0, marker='o', c='r', s=100)
        ax.scatter(5, 0, 0, marker='o', c='r', s=100)
        ax.scatter(5, 5, 0, marker='o', c='b', s=100)
        plt.title('The Hyper-plane after Applying Sigmoid()')
        plt.xlabel('x1', fontsize=20)
        plt.ylabel('x2', fontsize=20)
        plt.grid()
        plt.show()
    # Most e tartalmazza az osszes hibat a training alatt. Atlagoljuk es hozzadjuk a E[]-hez
    E.append(np.mean(e))
