import numpy as _np
import matplotlib as _plt


def zplane(zeros=[], poles=[], roc=[], radii=0, offsets=[], axis=True):
    '''Dibujar el diagrama de polos y zeros
    '''
    # por si acaso
    zeros = _np.atleast_1d(zeros)
    poles = _np.atleast_1d(poles)
    roc = _np.atleast_1d(roc)

    # get a figure/plot
    ax = _plt.pyplot.gca()

    # Scale axes to fit
    zp = zeros
    zp = _np.append(zp, poles)
    zpr = _np.real(zp)
    zpi = _np.imag(zp)
    if len(offsets) == 0:
        xoffset = 0.5
        yoffset = 0.15
    elif len(offsets) == 1:
        xoffset = offsets[0]
        yoffset = offsets[0]
    else:
        xoffset = offsets[0]
        yoffset = offsets[1]
    xmin = _np.amin(_np.append(zpr, [-1])) - xoffset
    xmax = _np.amax(_np.append(zpr, [+1])) + xoffset
    ymin = _np.amin(_np.append(zpi, [-1])) - yoffset
    ymax = _np.amax(_np.append(zpi, [+1])) + yoffset
    ax.axis('scaled')
    ax.axis([xmin, xmax, ymin, ymax])

    # ROC
    if len(roc) != 0:
        if len(roc) == 1:
            roc = _np.concatenate([roc, [0]])
        roc = _np.sort(_np.abs(roc))
        gray = '0.85'
        if roc[1] == _np.inf:
            ax.add_patch(_plt.patches.Rectangle((xmin, ymin), xmax-xmin,
                                                ymax-ymin, fill=True,
                                                color=gray))
        else:
            ax.add_patch(_plt.patches.Circle((0, 0), radius=roc[1],
                                             fill=True, color=gray))
            ax.add_patch(_plt.patches.Circle((0, 0), radius=roc[1],
                                             fill=False, color='black',
                                             ls='dotted', alpha=0.8))
        if roc[0] != 0:
            ax.add_patch(_plt.patches.Circle((0, 0), radius=roc[0],
                                             fill=True, color='white',
                                             alpha=1))
            ax.add_patch(_plt.patches.Circle((0, 0), radius=roc[0],
                                             fill=False, color='black',
                                             ls='dotted', alpha=0.8))

    # Add unit circle
    unit_circle = _plt.patches.Circle((0, 0), radius=1, fill=False, lw=1,
                                      color='black', ls='solid', alpha=1)
    ax.add_patch(unit_circle)

    # ejes
    if axis:
        ax.minorticks_on()
        ax.set_xlabel('Parte real')
        ax.set_ylabel('Parte imaginaria')
    else:
        ax.axis('off')

    ax.axvline(0, color='black', alpha=0.5)
    ax.axhline(0, color='black', alpha=0.5)
    if radii != 0:
        for r in _np.arange(radii, 180, radii):
            talpha = _np.tan(_np.deg2rad(r))
            ax.add_line(_plt.lines.Line2D([xmin, xmax],
                                          [xmin*talpha, xmax*talpha],
                                          color='black', lw=0.5, ls='dotted'))

    # plt.text(0.05, ymax-0.11, r'$\mathfrak{Im}$', fontsize=18)
    # plt.text(xmax - 0.12, 0.05, r'$\mathfrak{Re}$', fontsize=18)

    # Plot the zeros and set marker properties
    ax.plot(zeros.real, zeros.imag, 'o', markersize=8, mec='black', mew=1,
            mfc='white', fillstyle='full')
    # Plot the poles and set marker properties
    ax.plot(poles.real, poles.imag, 'x', markersize=8, mec='black', mew=1)

    # Ticks X
    # xticks = _np.unique(zpr)
    # for xt in xticks:
    #    if xt == 0:
    #        continue
    #    #y = -0.12;
    #    #if xt < 0:
    #    #    x = xt - 0.12
    #    #else:
    #    #    x = xt + 0.08
    #    x = xt
    #    y = -0.12
    #    xf = Fraction(xt).limit_denominator(10)
    #    if xf.denominator != 1:
    #        ax.text(x, y, r'${{{}}}/{{{}}}$'.format(xf.numerator,
    #                xf.denominator), fontsize=14)
    #    else:
    #        ax.text(x, y, r'${{{}}}$'.format(xf.numerator), fontsize=10)

    return ax
