

class test:
    def __init__(self):
        self.t = 3
    def plus(self):
        self.t += 1

a = test()
b = test()

c = a

c.plus()

print(a.t, b.t, c.t)

c = b

c.plus()
print(a.t, b.t, c.t)