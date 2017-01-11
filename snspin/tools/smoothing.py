#/bin/env python

import scipy as S
import numpy as N
import pylab as P
try:
    from scipy import factorial
except:
    from scipy.misc import factorial  # To handle different version of scipy


# generic functions for the prediction

def B_matrix_from_A(A, W):
    """ B matrix is the smoothing matrix, I.E Y(reconstructed)=B Y(predicted)
    A matrix is such that AX=Y
    W is the weight matrix """
    covX = N.linalg.inv(N.dot(N.dot(N.transpose(A), W), A))
    return N.dot(N.dot(A, covX), N.dot(N.transpose(A), W))


def prediction_error(r, B, var, W=None, verbose=False):
    """ r is the vector of residuals
    B is the B matrix (see above)
    furthemore, we assume the error is pure variance (var)
    r is the residual vector """
    # simplification because we have no covariance included
    if W == None:
        pe = N.sum(r**2 / var) + 2 * N.sum(N.diag(B)) - len(var)
        if pe < 0 and verbose:
            print "WARNING <prediction_error>: pe < 0, variance probably under estimated"
    else:
        pe = N.dot(N.dot(r, W), r) + 2 * N.sum(N.diag(B)) - len(var)
        if pe < 0 and verbose:
            print "WARNING <prediction_error>: pe < 0, variance probably under estimated"
    return pe

# application to splines


def spline_A_matrix(x, tck, der=0):
    """ A is defined by Ax=y where x are the parameters and y the observable """
    t = tck[0]
    k = tck[2]
    A = N.zeros([len(x), len(tck[1]) - 4])
    for i in xrange(len(tck[1]) - 4):
        c = N.zeros(len(tck[1]))
        c[i] = 1
        y = S.interpolate.splev(x, (t, c, k), der)
        A[:, i] = y
    return A


def spline_tck_prediction_error(x, y, var, tck, dbg=False, Wcorr=None):
    A = spline_A_matrix(x, tck)
    out = S.interpolate.splev(x, tck)
    r = out - y
    # here by construct, the spline uses only the diagonal term on W for
    # minimization : this is NOT the Wcorr ...
    W = N.diag(1. / var)
    B = B_matrix_from_A(A, W)
    e = prediction_error(r, B, var, Wcorr)
    return out, tck, e


def spline_prediction_error(x, y, var, s, dbg=False, W=None):
    tck = S.interpolate.splrep(x, y, 1 / N.sqrt(var), s=s)
    out = spline_tck_prediction_error(x, y, var, tck, Wcorr=W)
    if dbg:
        print s, out[2]
    return out


def weight_matrix_corr(var, corr):
    # computation of the weight matrix :
    V = N.diag([1.] * len(var))
    V += N.diag([corr] * (len(var) - 1), 1)
    V += N.diag([corr] * (len(var) - 1), -1)
    W = N.linalg.inv(V * N.transpose([N.sqrt(var)]) * N.sqrt(var))
    return W


def spline_find_s(x, data, var, s0=None, corr=0.0, dbg=False):
    if s0 == None:
        s = len(x) * 1.
    else:
        s = s0 * 1.
    # we added an enforcment of s>0
    s_tweak = len(x)**2 * 0.04
    # computation of the weight matrix :
    if corr != 0.0:
        V = N.diag([1.] * len(var))
        V += N.diag([corr] * (len(var) - 1), 1)
        V += N.diag([corr] * (len(var) - 1), -1)
        W = weight_matrix_corr(var, corr)
        func = lambda s, x, d, v: spline_prediction_error(
            x, d, v, (s + N.sqrt(s**2 + s_tweak)) / 2., dbg=dbg, W=W)[2]
    else:
        func = lambda s, x, d, v: spline_prediction_error(
            x, d, v, (s + N.sqrt(s**2 + s_tweak)) / 2., dbg=dbg)[2]

    #s = optimize.fmin(func, s, (x, data, var), xtol=s/100.)
    s = gauss_newton(func, s, (x, data, var), xtol=s / 100.)
    return (s + N.sqrt(s**2 + s_tweak)) / 2.

# Other tools for splines


def spline_covariance(x, var, tck, A=None):
    """ returns the covariance matrix of spline parameters """
    if A == None:
        A = spline_A_matrix(x, tck)
    W = N.diag(1. / var)
    return N.linalg.inv(N.dot(N.dot(N.transpose(A), W), A))


def spline_after_fit(x, y, var, tck, covX=None, A=None):
    """ returns the spline parameters as a linear fit for a given tck """
    if A == None:
        A = spline_A_matrix(x, tck)
    W = N.diag(1. / var)
    if covX == None:
        covX = spline_covariance(x, var, tck, A)
    X = N.dot(N.dot(covX, N.transpose(A)), N.dot(W, y))
    return X


def derivatives_covariance(x, tck, CovX):
    """ x is a coordinate (wavalength) near which we want to make a limited developpment
    Par are the parameters of the derivative centered in x : 0"""
    D = N.zeros([3, len(tck[1]) - 4])
    D[0, :] = spline_A_matrix([x], tck, 1)
    D[1, :] = spline_A_matrix([x], tck, 2)
    D[2, :] = spline_A_matrix([x], tck, 3)
    Par = N.dot(D, tck[1][:-4])
    CovD = N.dot(N.dot(D, CovX), N.transpose(D))
    return Par, CovD


# specific case of the minimum
def discriminant(par):
    a = par[2]
    b = par[1]
    c = par[0]
    return b**2 - 2 * a * c


def solution(par):
    a = par[2]
    b = par[1]
    c = par[0]
    delta = discriminant(par)
    xm = (-b - N.sqrt(delta)) / a
    xp = (-b + N.sqrt(delta)) / a
    # choose the solution closer to the extected value/
    if abs(xm) < abs(xp):
        e = -1
        x = xm
    else:
        e = 1
        x = xp
    return x, e


def gradpar(par, e):
    a = par[2]
    b = par[1]
    c = par[0]
    delta = discriminant(par)
    dA = b / a**2 + e * (c * a - b**2) / (N.sqrt(delta) * a**2)
    dB = - 1 / a + e * b / N.sqrt(delta) / a
    dC = -e / N.sqrt(delta)
    # to keep the order of the parameters
    return N.array([dC, dB, dA])


def secondpar(par, e):
    a = par[2]
    b = par[1]
    c = par[0]
    delta = discriminant(par)
    dAdA = - (e * b**4 - 6 * e * b**2 * delta - 3 * e * delta **
              2 + 8 * delta**(1.5) * b) / (8 * delta**1.5 * a**3)
    dBdB = - e * c / delta**1.5
    dCdC = - e * a / (2 * delta**1.5)
    dAdB = (e * b**3 - 3 * e * b * delta + 2 *
            delta**1.5) / (2 * delta**1.5 * a**2)
    dAdC = -e * c / delta**1.5
    dBdC = e * b / (delta**1.5)
    # to keep the order of the parameters
    return N.array([[dCdC, dBdC / 2, dAdC / 2], [dBdC / 2, dBdB, dAdB / 2], [dAdC / 2, dAdB / 2, dAdA]])


def bias_var(covD, Grad, Second):
    bias = N.sum(Second * covD)
    var = N.sum(N.outer(Grad, Grad) * covD)
    return bias, var


def extremum_tck_bias_var(x0, x, var, tck):
    """ returns the bias and variance of an extremum near a given coordinate
        x0 is the location of the extremum
        x is the coordinates vector (lambda for instance)
        var is the variance vector of the data
        """
    covX = spline_covariance(x, var, tck)
    par, covD = derivatives_covariance(x0, tck, covX)
    x0, e = solution(par)
    G = gradpar(par, e)
    S = secondpar(par, e)
    return bias_var(covD, G, S)


def alltogether(x, y, sigma, x0, nloop):
    """ x and y are the original coordinates of figure"""
    xtrm_l = []
    bias_l = []
    var_l = []
    for i in xrange(nloop):
        rda = N.random.normal(size=1000)
        new_y = y + rda * sigma
        var = N.array([sigma**2] * len(x))
        tck = S.interpolate.splrep(x, new_y, 1. / N.sqrt(var), s=len(x) * 10)
        covX = spline_covariance(x, var, tck)
        par, covD = derivatives_covariance(x0, tck, covX)
        xtrm_rel, e = solution(par)
        G = gradpar(par, e)
        S = secondpar(par, e)
        bias, var = bias_var(covD, G, S)
        xtrm_l.append(xtrm_rel + x0)
        bias_l.append(bias)
        var_l.append(var)
    return xtrm_l, bias_l, var_l

# Gaussian filter in velocity space


def B_matrix_gauss_filter(spec, sigma):
    """ spec is the spectrum on which to compute the Gauss_Filter
    sigma is the delta/lambda function applied"""
    B = N.zeros([len(spec.x), len(spec.x)])
    for i in spec.index_list:
        B[:, i] = 1. / (spec.x * N.sqrt(2 * N.pi) * sigma) * \
            N.exp(-((N.log(spec.x) -
                     N.log(spec.x[i])) / sigma)**2 / 2.) * spec.step
        B[:, i] /= N.sum(B[:, i])
    return B


def gauss_prediction_error(spec, sigma, dbg=False):
    B = B_matrix_gauss_filter(spec, sigma)
    y = N.dot(B, spec.data)
    e = prediction_error(y - spec.data, B, spec.var)
    if dbg == True:
        print sigma, e
    return e


def gauss_find_sigma(spec):
    s = 0.001
    # we added an enforcment of s>0
    func = lambda s, spec: gauss_prediction_error(spec, s, dbg=True)
    s = gauss_newton(func, s, (spec, ))
    #s = optimize.fmin(func, s, (spec, ))
    return s


def gauss_newton(func, x, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, dinit=0.25):

    xgrid = x * N.arange(0, dinit * 7, dinit)
    fgrid = N.array([func(x, *args) for x in xgrid])

    index = fgrid.tolist().index(N.min(fgrid[1:-1]))
    xs = xgrid[index - 1:index + 2]
    fs = fgrid[index - 1:index + 2]

    """x1=x*(1+dinit)
    x2=x*(1-dinit)
    xs=array([x2, x, x1])
    fs=array([func(x2, *args), func(x, *args), func(x1, *args)])
"""
    iter = 3
    ierr = 0
    lbound = -N.inf
    ubound = +N.inf
    converged = 0
    while converged == 0 and ierr == 0:
        if fs[1] == N.min(fs):
            if xs[0] > lbound:
                lbound = xs[0]
            if xs[2] < ubound:
                ubound = xs[2]
        iter += 1
        m = N.array(
            [[xs[0]**2, xs[0], 1], [xs[1]**2, xs[1], 1], [xs[2]**2, xs[2], 1]])
        y = N.array([fs[0], fs[1], fs[2]])
        p = N.dot(N.linalg.inv(m), y)
        # print p
        if p[0] < 0 or (p[0] == 0 and p[1] != 0):
            # max, not min. make a gradient descent
            if fs[0] < fs[2]:
                # x<0 direction
                newx = xs[0] + 2 * (xs[0] - xs[2])
                if newx < lbound:
                    if xs[0] == lbound:
                        newx = (xs[1] + xs[0]) / 2
                    else:
                        newx = (xs[0] + lbound) / 2
                newf = func(newx, *args)
                ireplace = 2
            else:
                # x>0 direction
                newx = xs[2] + 2 * (xs[2] - xs[0])
                if newx > ubound:
                    if xs[2] == ubound:
                        newx = (xs[1] + xs[2]) / 2
                    else:
                        newx = (xs[2] + ubound) / 2
                newf = func(newx, *args)
                ireplace = 0

        elif p[0] > 0:
            # find coordinates of the minimum
            newx = -p[1] / 2 / p[0]
            if newx < lbound:
                if xs[0] == lbound:
                    newx = (xs[1] + xs[0]) / 2
                else:
                    newx = (xs[0] + lbound) / 2
            if newx > ubound:
                if xs[2] == ubound:
                    newx = (xs[1] + xs[2]) / 2
                else:
                    newx = (xs[2] + ubound) / 2
            newf = func(newx, *args)
            if N.min(fs) == fs[2]:
                ireplace = 0
            elif N.min(fs) == fs[0]:
                ireplace = 2
                # new x in inside
            else:
                if newx > xs[1]:
                    ireplace = 2
                else:
                    ireplace = 0
                if newf < fs[2]:
                    ireplace = 2 - ireplace
        else:
            # flat universe
            ierr = 1

        if maxiter != None and iter > maxiter:
            ierr = 2

        if ierr == 0:
            if N.min(N.abs(newx / xs - 1)) < xtol and N.min(N.abs(newf / fs - 1)) < ftol:
                converged = 1

            xs[ireplace] = newx
            fs[ireplace] = newf

            index = N.argsort(xs)
            fs = fs[index]
            xs = xs[index]
    print "Number of iterations " + str(iter)
    return xs[fs == N.min(fs)]

################## Savitzky-Golay ###########
# from pySNURP


def sg_coeff(num_points, pol_degree, diff_order=0):
    """
    calculates filter coefficients for symmetric savitzky-golay filter.

    see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

    num_points   means that 2*num_points+1 values contribute to the smoother.
    pol_degree   is degree of fitting polynomial
    diff_order   is degree of implicit differentiation.
                 0 means that filter results in smoothing of function
                 1 means that filter results in smoothing the first derivative of function.
                 and so on ...
    """

    # uses a slow but sure algorithm
    # setup normal matrix
    r = N.arange(-num_points, num_points + 1, dtype=float)
    A = N.array([r**i for i in range(pol_degree + 1)]).transpose()

    # calculate diff_order-th row of inv(A^T A)
    ATA = N.dot(A.transpose(), A)
    Am = N.linalg.inv(ATA)

    # calculate filter-coefficients
    coeff = N.dot(Am, N.transpose(A))[diff_order] * factorial(diff_order)

    return coeff


def B_matrix_sg(num_points, pol_degree, dim):
    coeffs = sg_coeff(num_points, pol_degree)
    B = N.diag([coeffs[num_points]] * dim)
    for i in xrange(num_points):
        B += N.diag([coeffs[num_points - 1 - i]] * (dim - 1 - i), i + 1)
        B += N.diag([coeffs[num_points - 1 - i]] * (dim - 1 - i), -i - 1)
    for i in xrange(dim):
        B[i, :] /= sum(B[i, :])
    return B


def sg_find_num_points(x, data, var, pol_degree=2, corr=0.0, verbose=False):
    """
    Returns the size of the halfwindow for the best savitzky-Golay approximation.
    """
    #+ FIXME: wouldn't it be better to explore by dichotomy between 1 and len(data)
    #+ instead ?  (SZF 2011-11-17)

    def n(i_iteration):
        return (i_iteration - 1) + pol_degree / 2

    W = weight_matrix_corr(var, corr)
    e = {}
    finished = False
    i_iteration = 1
    #- coarse exploration to estimate the number of points
    #- yielding the best smoothing, i.e. yielding the lower "prediction_error"
    #- defined as the error made by the smoothing fonction approximating the data
    #- taking into account the variance of the signal.
    while not finished:
        num_points = n(i_iteration)
        B = B_matrix_sg(num_points, pol_degree, len(x))
        yy = N.dot(B, data)
        pe = prediction_error(yy - data, B, var, W)
        if verbose:
            print '%d, %f' % (num_points, pe)
        e[num_points] = pe
        if (n(i_iteration * 2) > len(x)):
            # Test to prevent the coarse exploration to end up testing
            # number of points larger than the total number of data points.
            # The i_iteration *=2 takes into account that the next finer exporation expects
            # the coarse exploration to have ended one step *after* the
            # inflection point.
            i_iteration *= 2
            B = B_matrix_sg(len(x), pol_degree, len(x))
            yy = N.dot(B, data)
            pe = prediction_error(yy - data, B, var, W)
            e[len(x)] = pe
            finished = True
        elif ((pe > N.min(e.values()) and N.min(e.values()) < len(x) * 0.9999)):
            # Test to stop when the prediction error stops decreasing and starts increasing again.
            # Under the assumption that the prediction error is convex and starts by decreasing,
            # this means that the inflection point happened just before this
            # step.
            finished = True
        else:
            i_iteration *= 2
    if i_iteration < 3:
        return n(i_iteration)

    #- In the case where the previous exploration was stopped because n(i_iteration) > len(x)
    #- the last key of e is not n(i_iteration) but len(x)
    toler = N.max([e[n(i_iteration / 4)] - e[n(i_iteration / 2)],
                   e[min(len(x), n(i_iteration))] - e[n(i_iteration / 2)]]) / 2
    #- Finer exploration of the region between where we know the prediction error was decreasing and
    #- either where it was increasing again or the total number of data points
    for num_points in N.arange(n(i_iteration / 4),
                               min(n(i_iteration), len(x)),
                               max([1, n(i_iteration / 4) / 10])):
        B = B_matrix_sg(num_points, pol_degree, len(x))
        yy = N.dot(B, data)
        pe = prediction_error(yy - data, B, var, W)
        e[num_points] = pe
        if verbose:
            print '%d, %f' % (num_points, pe)
        if num_points > n(i_iteration / 2) and (pe - N.min(e.values()) > toler):
            break

    result = e.keys()[e.values().index(N.min(e.values()))]

    if result >= len(x):
        # This is to avoid the always problematic limit case of calculating an interpolator on
        # all the points available (for example, in the Savitzky-Golay interpolation
        # (ToolBox.Signal), when all the available points are selected, the convolution by the
        # kernel chops out the two extreme points).
        result = len(x) - 1
    return result

#####################################################
# Test the B matrix
#####################################################


def comp_epsilon_terms(x, y, v, p, corr=0, mode='sp', all=False):
    if mode == 'sg':
        B = B_matrix_sg(2 * p + 1, 2, len(x))
        yy = N.dot(B, y)
    elif mode == 'sp':
        tck = S.interpolate.splrep(x, y, 1 / N.sqrt(v), s=p)
        A = spline_A_matrix(x, tck)
        B = B_matrix_from_A(A, N.diag(1. / v))
        yy = S.interpolate.splev(x, tck)
    else:
        raise ValueError(
            'Error: mode should be either "sp" for spline or "sg" for savitzky-golay')
    W = weight_matrix_corr(v, corr)
    rchi2 = N.dot(N.dot(yy - y, W), yy - y)
    TB = N.trace(B)
    lv = len(v)
    if all:
        return rchi2, TB, lv, p, yy
    else:
        return rchi2, TB, lv, p


def plot_params_evol(rchi2, TB, lv, p):

    rchi2, TB, lv, p = map(N.array, [rchi2, TB, lv, p])

    fig = P.figure()
    ax = fig.add_subplot(111)
    ax.plot(p, rchi2, 'k', label='Regular chi2')
    ax.plot(p, rchi2 + TB - lv, 'r', label='Epsilon')
    ax.legend(loc='best')

##################################


def savitzky_golay(data, kernel=11, order=4, derivative=0):
    """Applies a Savitzky-Golay filter.

    :param array data: input numpy 1D-array
    :param int kernel: a positive odd integer > order+2 giving the kernel size
    :param int order: order of the polynomial
    :param int derivative: 1 or 2 for 1st or 2nd derivatives
    :return: smoothed data as a numpy array

    Source: http://www.scipy.org/Cookbook/SavitzkyGolay
    """

    try:
        kernel = abs(int(kernel))
        order = abs(int(order))
    except ValueError:
        raise ValueError("kernel and order must be castable to int")
    if kernel % 2 != 1 or kernel < 1:
        raise TypeError("kernel=%d is not a positive odd number" % kernel)
    if kernel < order + 2:
        raise TypeError("kernel=%d is not > order+2=%d" % (kernel, order + 2))

    hsize = (kernel - 1) // 2
    b = N.mat([[k**i for i in range(order + 1)]
               for k in range(-hsize, hsize + 1)])
    weights = N.linalg.pinv(b).A[derivative]
    hsize = (len(weights) - 1) // 2

    # Temporary data, extended with a mirror image to the left and right
    # left extension: f(x0-x) = f(x0)-(f(x)-f(x0)) = 2f(x0)-f(x)
    # right extension: f(xl+x) = f(xl)+(f(xl)-f(xl-x)) = 2f(xl)-f(xl-x)
    leftpad = 2 * data[0] - data[1:hsize + 1][::-1]
    rightpad = 2 * data[-1] - data[-(hsize + 1):-1][::-1]
    data = N.concatenate((leftpad, data, rightpad))

    # Convolution
    sdata = N.convolve(data, weights, mode='valid')

    return sdata
