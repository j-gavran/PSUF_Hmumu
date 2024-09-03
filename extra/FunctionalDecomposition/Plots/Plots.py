import itertools
import numpy                           as np
import matplotlib.pyplot               as plt
import matplotlib.tri                  as tri
import matplotlib.cm                   as cm

from   matplotlib                      import gridspec
from   matplotlib.backends.backend_pdf import PdfPages
from   math                            import sqrt, pi, ceil
from   scipy.stats                     import ks_2samp
from   scipy.special                   import erf

from   Tools.Decomp                    import TruncatedSeries

_show = False

# Integrate a function over bins as specified by edges. Npt points per bin.
def _integrate(func, edges, Npt=100):
    N = len(edges) - 1
    t = np.zeros((N,))

    x   = [ np.linspace(edges[n], edges[n+1], Npt + 1) for n in range (N) ]
    x   = np.asarray(x)
    s   = x.shape
    z   = func( x.flatten() ).reshape(s)

    for n in range(N):
        t[n] = np.trapz( z[n], dx = 1. / Npt)

    return t

# Gaussian CDF
def _cdf(z):
    return (1./2) * ( 1 - erf(z/sqrt(2)) )

# Interleave list items
def _flip(items, ncol):
    return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))

#########
# A plot decorator providing some common code for the remaining plots.
def gsPlot(*fargs, **fkw):
    def outer(func):
        def wrap(*args, **kwargs):
            fname = kwargs.pop("fname", None)
            pdf   = kwargs.pop("pdf", None)

            print fname

            fig = plt.figure()
            gs  = gridspec.GridSpec(*fargs, **fkw)

            # Run the function
            r   = func(fig, gs, *args, **kwargs)

            # Do some layout cleanup and save.
            gs.tight_layout(plt.gcf(), rect=[0, 0, 1, 0.97])
            gs.update(wspace=0.00, hspace=0.1)
            if pdf is not None:
                plt.savefig(pdf, format='pdf')
            if fname is not None:
                plt.savefig(fname)
            if _show:
                plt.show()

            plt.gcf().clear()

            return r
        return wrap
    return outer

######## CUTFLOW #######
@gsPlot(1, 1)
def cutflow(fig, gs, cutflow):
    ax  = [ plt.subplot(g) for g in gs ]

    cut, yld = zip(*cutflow)
    cuty     = np.arange(len(cut))[::-1]
    bar      = ax[0].barh(cuty, yld, align='center', alpha=0.5)

    fig.suptitle ('Cutflow')
    ax[0].set_yticks(cuty)
    ax[0].set_yticklabels(cut)
    ax[0].set_xlabel('Yield (Events)')

    # Annotate
    for h, b in zip(yld, bar):
        bx = h + 0.01*max(yld)
        by = b.get_y() + b.get_height()/2.

        ax[0].text(bx, by, "%.1f" % h, ha='left', va='center')

    ax[0].margins(x=0.175)
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

######## HYPERPARMETER SCAN #######
@gsPlot(1, 2, width_ratios=[19, 1])
def scan(fig, gs, L, A, LLH, LBest, Ld, Ad, fin, maxZ=200, points=False):
    ax  = [ plt.subplot(g) for g in gs ]

    # interpolate between data points for contouring
    triI   = tri.Triangulation(L.flatten(), A.flatten())
    ref    = tri.UniformTriRefiner(triI)

    dLLH   = np.minimum(LLH - LBest, 1.5*maxZ)
    triO   = ref.refine_field(dLLH.flatten(), subdiv=3)

    cmap   = cm.get_cmap(name='terrain', lut=None)
    levels = np.linspace(0, maxZ, 51)
    colors = [ '0.00', '0.25', '0.25', '0.25', '0.25']
    lws    = [   0.40,   0.25,   0.25,   0.25,   0.25]

    Csf    = ax[0].tricontourf(*triO, levels=levels, cmap=cmap)
    Cs     = ax[0].tricontour (*triO, levels=levels, colors=colors, linewidths=lws)
    if points:
        sca = ax[0].scatter(L,  A,  marker='.', s= 1.0, color='k', label="Scan Point")
    sca2   = ax[0].scatter(Ld, Ad, marker='o', s=20.0, color='r', label="Initial", zorder=10)
    sca3   = ax[0].scatter(*fin,   marker='x', s=35.0, color='r', label="Final",   zorder=10)

    fig.colorbar(Csf, ticks=levels[::5], cax=ax[1])

    fig.suptitle("Hyperparameter Scan")
    ax[0].legend(loc='upper left')
    ax[0].set_xlabel(r'Scale ($\lambda$)')
    ax[0].set_ylabel(r'Exponent ($\alpha$)')
    ax[1].set_ylabel(r'$\Delta$ Log-Likelihood')

######## PLOT #######
@gsPlot(3, 1, height_ratios=[3, 1, 1] )
def fit(fig, gs, D, **kwargs):
    ax  = [ plt.subplot(g) for g in gs ]

    # Parameters
    Bins     = kwargs.get("Bins")
    Title    = kwargs.get("Title",  "Unnamed")
    XLabel   = kwargs.get("XLabel", "Mass")
    YLabel   = kwargs.get("YLabel", "Events / Bin")
    LogX     = kwargs.get("LogX",   True)
    LogY     = kwargs.get("LogY",   True)
    Style    = kwargs.get("Style",  "bar")
    ResYLim  = kwargs.get("ResYLim", (-2.5, 2.5))
    YLim     = kwargs.get("YLim",    None)

    # Get some bin-derived quantities
    ctr      = (Bins[1:] + Bins[:-1])/2
    wd       = np.diff(Bins)
    rn       = (Bins[0], Bins[-1])

    h, _     = np.histogram(D.x, bins=Bins, range=rn, weights=D.w)
    h       *= D.Nint / wd
    t        = np.linspace(rn[0], rn[1], 50*len(Bins))
    err      = np.sqrt(h*wd, dtype=np.double)/wd

    # Make fit comparison
    tb       = D.Nint * _integrate(D.TestB, Bins)
    ts=0
    if len(D.GetActive()) > 0:
        ts       = D.Nint * _integrate(D.TestS, Bins)
        res      = (h*wd - ts*wd)/np.sqrt(ts*wd)
    else:
        res      = (h*wd - tb*wd)/np.sqrt(tb*wd)

    # Histogram and fit
    if Style == "bar":
        ax[0].bar(ctr, h, width=wd, log=LogX, label='Data', edgecolor="none", lw=0)
    elif Style == "errorbar":
        ax[0].errorbar(ctr, h, xerr=wd/2, yerr=err, label='Data', color='k', fmt='o')
    else:
        print "Style key must be 'bar' or 'errorbar'."

    ax[0].plot(t, D.Nint*D.TestB(t), ls='--', color='red', label='Background', zorder=10)
    if len(D.GetActive()) > 0:
        ax[0].plot(t, D.Nint*D.TestS(t), ls='-',  color='red',   label='Signal+Bkg', zorder=10)
    ax[0].legend()
    ax[0].yaxis.grid(ls=':')
    ax[0].set_ylabel(YLabel)
    if LogY: ax[0].set_yscale('log')

    if YLim is not None: ax[0].set_ylim(*YLim)

    # The background-subtracted data
    if Style == "bar":
        ax[1].bar (ctr, h-tb, width=wd, edgecolor="none", lw=0)
    elif Style == "errorbar":
        ax[1].errorbar (ctr, h-tb, xerr=wd/2, yerr=err, color='k', fmt='o')
    else:
        print "Style key must be 'bar' or 'errorbar'."

    ax[1].plot(t, np.zeros_like(t), ls='--', color='red')
    if len(D.GetActive()) > 0:
        ax[1].plot(t, D.Nint*(D.TestS(t) - D.TestB(t)), ls='-', color='red', zorder=10)
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
    ax[1].set_ylabel(r'Data - Bkg')

    # The residual plot.
    ax[2].bar(ctr, res,  width=wd,               edgecolor="none", lw=0)
    ax[2].plot(t, 0*t, color='black', lw=1.0)
    ax[2].set_ylim(*ResYLim)
    ax[2].set_ylabel(r'Residual ($\sigma$)')

    # Shared formatting
    fig.suptitle(Title)

    for a in ax:
        a.xaxis.grid(ls=':')
        a.yaxis.set_label_coords(-0.065, 0.5)
        a.set_xlim(*rn)

        if LogX: a.set_xscale('log')
    for a in ax[:-1]:
        a.tick_params(labelbottom='off')
    ax[-1].set_xlabel(XLabel)

    return h, res

######## PULL ########
@gsPlot(1, 1)
def pull(fig, gs, data, res):
    ax          = [ plt.subplot(g) for g in gs ]

    kres        = np.compress(data > 20, np.nan_to_num(res))
    hist, edges = np.histogram(kres, bins=np.linspace(-5, 5, 21))
    centers     = (edges[1:] + edges[:-1])/2

    nrm         = np.random.normal(size=20*len(kres))
    ks_p        = ks_2samp(nrm, kres)[1] if len(kres) > 0 else 1.0

    fig.suptitle("Pull Distribution (Bins with $>20$ Events)")
    ax[0].set_xlabel(r'Deviation ($\sigma$)')
    ax[0].set_ylabel("Number of Bins")
    ax[0].bar     (centers, hist, width=np.diff(edges), label='Bin Residuals \n $p=%.2g$ (KS)' % ks_p)
    ax[0].errorbar(centers, hist, yerr=np.sqrt(hist), color='k', fmt='o')
    ax[0].set_xlim(-5, 5)

    t = np.linspace(-5, 5, 201)
    n = np.exp( -0.5*t**2 ) / sqrt(2*pi)
    ax[0].plot(t, 0.5*n*hist.sum(), lw=1.5, color='b', label=r'Standard Normal')
    ax[0].legend()

######## SIGNALS AND ESTIMATORS ########
@gsPlot(1, 1)
def estimators(fig, gs, D, **kwargs):
    ax      = [ plt.subplot(g) for g in gs ]

    # Parameters
    Signals = kwargs.get("Signals",    [])
    Draw    = set(kwargs.get("Draw",   ["Estimators"]))

    Range   = kwargs.get("Range")
    Title   = kwargs.get("Title",  "Unnamed")
    XLabel  = kwargs.get("XLabel", "Mass")
    YLabel  = kwargs.get("YLabel", "Arbitrary Units")
    LogX    = kwargs.get("LogX",   True)

    t       = np.linspace(Range[0], Range[1], 1001)

    ax[0].plot(t, 0*t, lw=0.75, color='k')
    for sigName in Signals:
       eName = sigName.replace('%', '\%')
       M     = np.zeros_like(D[sigName].Sig)

       if "Signal" in Draw:
           M[:]    = D[sigName].Sig
           E       = TruncatedSeries(D.Factory, M)
           ax[0].plot(t, E(t), lw=1.0, label=eName + " (Signal)")

       if "Residual" in Draw:
           M[:D.N] = 0
           M[D.N:] = D[sigName].Res
           E       = TruncatedSeries(D.Factory, M)
           ax[0].plot(t, E(t), lw=1.0, label=eName + " (Residual)")

       if "Estimator" in Draw:
           M[:D.N] = 0
           M[D.N:] = D[sigName].Est
           E       = TruncatedSeries(D.Factory, M)
           ax[0].plot(t, E(t), lw=1.0, label=eName + " (MinVar Estimator)")

    fig.suptitle(Title)
    ax[0].set_xlabel(XLabel)
    ax[0].set_ylabel(YLabel)
    ax[0].set_xlim(*Range)
    ax[0].legend()

######## MOMENT LINE/BAR PLOT ########
@gsPlot(1, 1)
def moments(fig, gs, D, **kwargs):
    def _bplot(a, x, y, label, style, Num, n):
        if style == "line":
            a.plot(x, y**2, label=label)
        elif style == "bar":
            a.bar (Num*x + n, y**2, label=label, lw=0)

    ax         = [ plt.subplot(g) for g in gs ]

    # Parameters
    Signals    = kwargs.get("Signals",    [])
    Draw       = set(kwargs.get("Draw",   ["Estimators"]))

    Range      = kwargs.get("Range")
    Style      = kwargs.get("Style",  "line")
    Title      = kwargs.get("Title",  "Unnamed")
    XLabel     = kwargs.get("XLabel", "Moment #")
    YLabel     = kwargs.get("YLabel", r"$\left|\mathrm{Moment}\right|^2$")
    LogX       = kwargs.get("LogX",   True)
    LogY       = kwargs.get("LogY",   True)

    ctr        = np.arange(*Range)
    Num        = 2 + len(Draw) * len(Signals)

    _bplot( ax[0], ctr, D.Mom[ctr], "Data", Style, Num, 0)
    n = 1

    for sigName in Signals:
        eName = sigName.replace('%', '\%')
        M     = np.zeros_like(D[sigName].Sig)

        if "Signal" in Draw:
            M[:] = D[sigName].Sig
            _bplot(ax[0], ctr, M[ctr], eName + " (Signal)",           Style, Num, n)
            n += 1

        if "Residual" in Draw:
            M[:D.N] = 0
            M[D.N:] = D[sigName].Res
            _bplot(ax[0], ctr, M[ctr], eName + " (Residual)",         Style, Num, n)
            n += 1

        if "Estimator" in Draw:
            M[:D.N] = 0
            M[D.N:] = D[sigName].Est
            _bplot(ax[0], ctr, M[ctr], eName + " (MinVar Estimator)", Style, Num, n)
            n += 1

    fig.suptitle(Title)
    ax[0].set_xlabel(XLabel)
    ax[0].set_ylabel(YLabel)
    ax[0].set_xlim(*Range)
    if LogY:
        ax[0].set_yscale('log')
    if Style == "bar":
        tnum = (int(Range[1]) / 8) * np.arange(9)
        ax[0].set_xticks     ( [(x*Num + Num/2) for x in tnum ])
        ax[0].set_xticklabels( [str(x)          for x in tnum ])
    elif Style == "line":
        tnum = (int(Range[1]) / 8) * np.arange(9)
        ax[0].set_xticks     ( [x               for x in tnum ])
        ax[0].set_xticklabels( [str(x)          for x in tnum ])

    ax[0].legend()

######## MASS SCAN ########
@gsPlot(3, 1, height_ratios=[6, 2, 2])
def mass_scan(fig, gs, scan, **kwargs):
    ax  = [ plt.subplot(g) for g in gs ]

    Units  = kwargs.get("Units",  "Events")
    Title  = kwargs.get("Title",  "Scan")
    XLabel = kwargs.get("XLabel", "Mass")
    LogX   = kwargs.get("LogX",   True)
    CMap   = kwargs.get("CMap",   "copper")
    Scans  = kwargs.get("Scans",  scan.keys())
    Bands  = kwargs.get("Bands",  len(Scans) == 1)
    XRange = kwargs.get("XRange",
               ( min([x.Mass.min() for x in scan.values()]),
                 max([x.Mass.max() for x in scan.values()]) ))

    sig    = { n: x.Sig                                     for n, x in scan.items() }
    keep   = { n: (x.Mass > XRange[0])*(x.Mass < XRange[1]) for n, x in scan.items() }

    sigmax =   max([x[keep[name]].max() for name, x in sig.items()])
    zero   = np.asarray((0,0))

    cmap   = [ plt.get_cmap(CMap)(i) for i in np.linspace(0, 1, len(Scans)) ]

    # Limits
    for c, name in zip(cmap, Scans):
        e = scan[name].ExpLim
        n = name.replace("%", "\%")

        if Bands:
            ax[0].fill_between(scan[name].Mass, e[:,0], e[:,4], color='y' )
            ax[0].fill_between(scan[name].Mass, e[:,1], e[:,3], color='g' )
        ax[0].plot(scan[name].Mass, e[:,2],            label=n, color=c, ls='--')
        ax[0].plot(scan[name].Mass, scan[name].ObsLim, label=n + " ", color=c, ls='-')

    extr = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    h, l = ax[0].get_legend_handles_labels()
    ax[0].legend(_flip([ extr,  extr] + h, 2),
                 _flip(["Exp", "Obs"] + l, 2),
                 loc='best', ncol=2)
    ax[0].set_ylabel(r'CL$_{95}$ (%s)' % Units)
    ax[0].set_yscale('log')

    # Deviation (sigmas)
    ax[1].fill_between(XRange, zero-2, zero+2, color='y' )
    ax[1].fill_between(XRange, zero-1, zero+1, color='g' )
    ax[1].plot        (XRange, zero,           color='k', ls=':' )
    for c, name in zip(cmap, Scans):
        ax[1].plot(scan[name].Mass, scan[name].Sig, label=name, color=c)

    ax[1].set_ylabel(r'Dev. ($\sigma$)')
    ax[1].set_ylim(-3, 3)

    # p-value
    for n in np.arange( 1, ceil(sigmax) + 1 ):
        t = 1.001*XRange[1] - 0.001*XRange[0]
        ax[2].plot(XRange, zero + _cdf(n), ls=':', lw=0.5,color='k')
        ax[2].text(t, _cdf(n), '$%d\sigma$' % n, va='center')
    for c, name in zip(cmap, Scans):
        M = scan[name].Mass  [keep[name]]
        P = scan[name].PValue[keep[name]]

        ax[2].plot(M, P, label=name, color=c)

    ax[2].set_ylabel(r'p-value')
    ax[2].set_yscale('log')

    # Shared formatting
    fig.suptitle(Title)

    for a in ax:
        a.set_xlim( *XRange )

        if LogX: a.set_xscale('log')
    for a in ax[:-1]:
        a.tick_params(labelbottom='off')
    ax[-1].set_xlabel(XLabel)

######## COEFFICIENT TABLES ########
@gsPlot(2, 3, width_ratios=[2, 1, 1])
def summary_table(fig, gs,  D):
    ax     = [ plt.subplot(gs[0,0]),
               plt.subplot(gs[1,0]),
               plt.subplot(gs[ :,2]) ]

    labels = D.GetActive()
    yields = [ "%.1f" % D[n].Yield  for n in labels ]
    uncs   = [ "%.1f" % D[n].Unc    for n in labels ]

    # Signals yields
    if len(labels) > 0:
        colL   = [ "Yield", "Uncertainty" ]
        txt    = zip(yields, uncs)

        ax[0].axis('tight')
        ax[0].axis('off')
        ax[0].set_title("Extracted Signal")
        ax[0].table(cellText=txt, rowLabels=labels, colLabels=colL, loc='center')

    # Correlations
    if len(labels) > 0:
        txt   = [ [ "%.3f" % D.Corr[n,m] if n<= m else "" for n in range(len(labels)) ] for m in range(len(labels)) ]
        ax[1].axis('tight')
        ax[1].axis('off')
        ax[1].set_title("Signal Correlations")
        ax[1].table(cellText=txt, rowLabels=labels, colLabels=labels, loc='center')

    # Moments
    colL  = [ "Value" ]

    rowL  = [ r'$\lambda$', r'$\alpha$' ]
    rowL += [ r'$c_{%d}$' % n              for n in range(D.TestB.Nmax) ]

    txt   = [ ["%.2f" % D.Factory[x]]      for x in ("Lambda", "Alpha") ]
    txt  += [ ["%.2g" % D.TestB.MomAct[n]] for n in range(D.TestB.Nmax) ]

    ax[2].axis('tight')
    ax[2].axis('off')
    ax[2].set_title("Background Moments")
    tab = ax[2].table(cellText=txt, rowLabels=rowL, colLabels=colL, loc='center')

