class Base(object):
    def __init__(self, x):
        self.x = x
    
    def func(self, f):
        self.x = f(f(self.x)) if isinstance(self, Twice) else f(self.x)
        print(self.x)
    
class Twice(Base):
    pass

double = lambda i: i*2
a = Base(5)
a.func(double)

b = Twice(5)
b.func(double)
