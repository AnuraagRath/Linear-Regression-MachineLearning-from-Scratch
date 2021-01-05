import seaborn
import matplotlib.pyplot as plt

def getGradientForB(x,y,b,m):
    N = len(x)
    diff = 0
    for i in range(N):
        xVal = x[i]
        yVal = y[i]
        diff += (yVal - ((m * xVal)+ b))
    bGradient = -(2/N) * diff
    return bGradient

def getGradientForM(x,y,b,m):
    N = len(x)
    diff = 0
    for i in range(N):
        xVal = x[i]
        yVal = y[i]
        diff += xVal * (yVal - ((m * xVal) + b))
    mGradient = -(2/N) * diff
    return mGradient

def stepGradient(bCurrent, mCurrent, x, y, learningRate):
    bGradient = getGradientForB(x, y, bCurrent, mCurrent)
    mGradient = getGradientForM(x, y, bCurrent, mCurrent)
    b = bCurrent - (learningRate * bGradient)
    m = mCurrent - (learningRate * mGradient)
    return [b, m]

def gradientDescent(x, y, learningRate, numIterations):
    b = 0
    m = 0
    for i in range(numIterations):
        b, m = stepGradient(b, m, x, y, learningRate)
    return [b, m]

#Data 
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

b, m = gradientDescent(months, revenue, 0.01, 1000)

y = [m*x + b for x in months]

plt.plot(months, revenue, "o")
plt.plot(months, y)

plt.show()