from PyQt5 import uic
import glob

for fname in glob.glob("dialogs/*.ui", recursive=True):
    print("converting", fname)
    fin = open(fname, 'r')
    fout = open(fname.replace(".ui", ".py"), 'w')
    uic.compileUi(fin, fout, execute=True)
    fin.close()
    fout.close()

for fname in glob.glob("Widgets/*.ui", recursive=True):
    print("converting", fname)
    fin = open(fname, 'r')
    fout = open(fname.replace(".ui", ".py"), 'w')
    uic.compileUi(fin, fout, execute=True)
    fin.close()
    fout.close()

for fname in glob.glob("*.ui", recursive=True):
    print("converting", fname)
    fin = open(fname, 'r')
    fout = open(fname.replace(".ui", ".py"), 'w')
    uic.compileUi(fin, fout, execute=True)
    fin.close()
    fout.close()