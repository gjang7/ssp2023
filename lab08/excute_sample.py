import vcodec4 as vc

c1 = vc.Config()
c1.set(32000)
print(c1)
c2 = vc.Config()
c2.set(44100,0.01,10,None)
print(c2)
c3 = vc.Config()
c3.set(44100,0.015,15,None)
print(c3)
c4=c3
c4.readfile("default.conf")
print(c4)
print(c3)
c5 = vc.Config()
c5.readfile("c2.conf")
print(c5)

fm = vc.FilterMemory()
fm.alloc(14,212)
print(fm)
fm.clear()
print(fm)

