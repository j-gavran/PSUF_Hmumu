'''
PrintMgr: functions for easy, consistent status printing to the screen with
ANSI colors.

PrintMgr is intended to print neatly-formatted status blocks to the screen,
which are periodically updated (e.g. current fit parameters during a fit).


pnew               : start a new status block
pstage             : start a new status block with a nice header
                     this is a decorator, and the decorated function is
                     expected to do its own printing.

pini / pdot / pend : new dotline / print dot / end dotline
pstr               : new status string
prst               : reset to beginning of block

'''

import sys, time

#############
_verbose  = True
_Ngap     = 32            # Number of dots per line
_Ndot     = 0             # Number of calls to _pdot
_Nline    = 0             # number of lines (of dots) that have been printed
_Nint     = 1             # print a dot every Nint calls
_Tloc     = 54            # Column to print times
_Ti       = time.time()   # Starting time.

_str_bl  = "\x1b[2K\r\x1b[1m%20s\x1b[0m: "                  # Emph (bold)
str_el   = "\x1b[2K\r\x1b[33;1m%20s\x1b[0m: "               # Emph (yellow)
str_nl   = "\x1b[2K\r%20s: "                                # Normal (white)
_str_sta = "\x1b[2K\r\x1b[1m[\x1b[32m %s \x1b[39m]\x1b[0m"  # Block header
_str_set = "\r\x1b[%dC"                                     # Set cursor to loc

#\x1b[2K\r == reset line

def _str_ar(pfx, ar, pfmt, afmt, plen):
    str  = pfmt % pfx
    for n, a in enumerate(ar):
        if n > 0 and n % 4 == 0:
            str += "\n" + " "*plen
        t = afmt % 0

        if a == 0:
          str += " "*(len(t)-2) + "- "
        elif not np.isfinite(a):
          u    = "%f " % a
          str += " "*(len(t)-len(u)) + u
        else:
          str += afmt % a
    return str

def pnew(str):
    global _Nline
    if not _verbose: return

    print str
    _Nline = 0

def pini(str, interval = 1):
    global _Ndot, _Nline, _Nint, _Ti
    if not _verbose: return
    
    str = _str_bl % str
    print str,
    sys.stdout.flush()
    _Ndot = 0
    _Nint = interval
    _Ti   = time.time()

def pdot(x=None, pchar="."):
    global _Ndot, _Nline
    if not _verbose: return

    _Ndot += 1
    q, r  = divmod(_Ndot, _Nint)

    if r != 0:
       return x

    sys.stdout.write(pchar)
    sys.stdout.flush()

    if q > 0 and q % _Ngap == 0:
        print " (%.2fs)" % (time.time() - _Ti)
        print " " * 22,
        _Nline += 1

    return x

def pstr(str):
    global _Ndot, _Nline
    if not _verbose: return

    print str
    _Nline += len(str.split("\n"))

def pend(str=""):
    global _Nline
    if not _verbose: return
    q, r  = divmod(_Ndot, _Nint)
    loc   = _Tloc - len(str) - 1

    if q > 0 and q % _Ngap == 0:
        return
    print _str_set % loc,
    print "%s (%.2fs)" % (str, time.time() - _Ti)
    _Nline += 1

    sys.stdout.flush()

def prst():
    global _Nline
    if not _verbose: return
    if _Nline == 0:  return
    print '\x1b[%dF' % _Nline,
    _Nline = 0

def pstage(str):
  def decorator(func):
      def wrap(*arg, **kw):
          verbose = kw.get("verbose", True)

          if verbose:
              t1 = time.time()
              pnew(_str_sta % str)
          r   = func(*arg, **kw)

          if verbose:
              t2  = time.time()
              print "Total time: %.2fs" % (t2 - t1)
          return r
      return wrap
  return decorator

