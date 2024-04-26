import numpy as np
import matplotlib.pyplot as plt
import _3Week
import _6Week

# Shift+F10을(를) 눌러 실행하거나 내 코드로 바꿉니다.
# 클래스, 파일, 도구 창, 액션 및 설정을 어디서나 검색하려면 Shift 두 번을(를) 누릅니다.
# 중단점을 전환하려면 Ctrl+F8을(를) 누릅니다.

def drawGraph_1():
    x = np.arange(0, 6, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

def drawGraph_2():
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) + np.cos(x)

    # 그래프그리기
    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, linestyle="--", label="cos")
    plt.plot(x, y3, label="sin+cos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('sin & cos & sin + cos')
    plt.legend()
    plt.show()

# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
     #drawGraph_2()

    # h = _3Week._3Week();
    # h.Test()

    # _3Week.XOR(0, 0)
    # _3Week.XOR(1, 0)
    # _3Week.XOR(0, 1)
    # _3Week.XOR(1, 1)
    #_6Week.sigmoidPrint()
    #_6Week.sigmoidPrint_2()
    #_6Week._6WeekDrawGraph()
    #_6Week.matrixMul()
    #_6Week._3floorNN()
    _6Week.softMax()