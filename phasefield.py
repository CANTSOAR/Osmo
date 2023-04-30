import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
import math
from math import sin, cos, pi

figures = 0
hold_graphs = False
#ax = plt.gca()
#ax.set_aspect('equal', adjustable='box')

def create_data(matrix = [[0, -1], [1, 0]], amount = 500, size = 5, circle = False):
  inputs = np.random.rand(2, amount) * (size * 2 + 1) - size
  matrix = np.array(matrix)

  if circle:
     inputs = np.array([[size * cos(360/amount * i * pi / 180) for i in range(amount)],
                        [size * sin(360/amount * i * pi / 180) for i in range(amount)]])

  outputs = np.dot(matrix, inputs)
  variance = (np.random.rand(2, amount) - .5) * size / 5

  outputs = outputs + variance

  x = inputs[0]
  y = inputs[1]
  u = outputs[0] - inputs[0]
  v = outputs[1] - inputs[1]

  return x, y, u, v

"""def create_data(amount = 500, size = 5, angle = 20):
  r = []
  theta = []
  
  for i in range(amount):
    r.append(random.random() * size)
    theta.append(random.random() * 360)
    
    r.append(r[-1] + size / 5 * (random.random() - .5))
    theta.append(theta[-1] + angle + angle/10 * random.random())

  x = []
  y = []
  u = []
  v = []
  
  for i in range(amount):
    x.append(r[2 * i] * cos(theta[2 * i] * pi / 180))
    y.append(r[2 * i] * sin(theta[2 * i] * pi / 180))

    u.append(r[2 * i + 1] * cos(theta[2 * i + 1] * pi / 180) - x[-1])
    v.append(r[2 * i + 1] * sin(theta[2 * i + 1] * pi / 180) - y[-1])

  return x, y, u, v"""

def plot_vectors(x, y, u, v, color = "cyan", show = False, newfig = True):
  if newfig:
    global figures
    figures += 1
    plt.figure(figures)
    plt.title("Plot of " + str(len(x)) + " Vectors")
  plt.quiver(*[x, y], *[u, v], scale=1, scale_units="xy", angles="xy", color = color, headwidth = 1, headlength = 1)
  if show and not hold_graphs:
    plt.show()

def color_gradient(points):
  colors = []

  for point in range(points):
    colors.append((point/points, 1, 1))

  return clrs.hsv_to_rgb(colors)

def errors(matrix, x, y, u, v, plot = False):
  matrix = np.array(matrix)
  
  inputs = np.array([x, y])
  estimates = np.dot(matrix, inputs) - inputs
  real_outputs = np.array([u, v])

  if plot:
    global figures
    figures += 1
    plt.figure(figures)
    plt.title("Error Plot")
    plot_vectors(x, y, estimates[0], estimates[1], "black", newfig = False)
    plot_vectors(x, y, u, v, "red", newfig = False)
    if not hold_graphs:
        plt.show()

  dots = np.diag(np.dot(estimates.T, real_outputs))

  lens = np.multiply(np.sqrt(np.diag(np.dot(estimates.T, estimates))), np.sqrt(np.diag(np.dot(real_outputs.T, real_outputs))))

  return 1 - np.mean(dots/lens)

def linearize(x, y, u, v, plot = False, adjust = True):
  thetapos = []
  thetadir = []
  
  for i in range(len(x)):
    thetapos.append((math.atan2(y[i], x[i]) * 180 / pi) % 360)
    thetadir.append((math.atan2(v[i], u[i]) * 180 / pi) % 360)
  
  if adjust:
    thetadir = (np.array(thetadir) - thetadir[thetapos.index(min(thetapos))]) % 360
  
  if plot:
    global figures
    figures += 1
    plt.figure(figures)
    plt.title("Linearized Vectors (Cartesian)")
    plt.scatter(thetapos, thetadir)
    if not hold_graphs:
        plt.show()

  return thetapos, list(thetadir)

def lsrl(linx, liny, plot = False):  
  linx = np.array(linx)
  liny = np.array(liny)

  sxx = np.sum((-linx + np.mean(linx))**2)
  sxy = np.dot(-linx + np.mean(linx), (-liny + np.mean(liny)).T)

  slope = sxy / sxx
  b = (liny.sum() - slope * linx.sum()) / len(linx)

  resid = np.array([abs(liny[i] - slope * linx[i] - b) for i in range(len(linx))])
  tempx = []
  tempy = []
  
  for i in reversed(range(len(linx))):
    if abs(liny[i] - slope * linx[i] - b) > 2 * np.mean(resid):
      tempx.append(linx[i])
      tempy.append(liny[i])
      linx = np.delete(linx, i)
      liny = np.delete(liny, i)

  sxx = np.sum((-linx + np.mean(linx))**2)
  sxy = np.dot(-linx + np.mean(linx), (-liny + np.mean(liny)).T)

  newslope = sxy / sxx
  newb = (liny.sum() - newslope * linx.sum()) / len(linx)

  resid = np.array([abs(liny[i] - slope * linx[i] - b) for i in range(len(linx))])
  
  if plot:
    global figures
    figures += 1
    plt.figure(figures)
    plt.title("LSRL (simple and revised)")
    plt.plot([min(linx), max(linx)], [min(linx) * slope + b, max(linx) * slope + b], color = "red")
    plt.plot([min(linx), max(linx)], [min(linx) * newslope + newb, max(linx) * newslope + newb], color = "cyan")
    plt.scatter(linx, liny)
    plt.scatter(tempx, tempy, color = "red")
    if not hold_graphs:
        plt.show()
  
  return newslope, newb, np.mean(resid)

def best_matrix(x, y, u, v, plot = False, eplot = False, eprint = False):
  x = np.array(x)
  y = np.array(y)
  u = np.array(u)
  v = np.array(v)

  lengths = (x**2 + y**2)**(1/2)
  
  loops = np.array([x + u, y + v]) / lengths

  lengths = []
  angles = []

  for a, b in loops.T:
    lengths.append(a**2 + b**2)
    angles.append(math.atan2(b, a) * 180 / pi % 360)

  loops = [[a for _, a in sorted(zip(angles, loops[0]))], [b for _, b in sorted(zip(angles, loops[1]))]]
  lengths = [i for _, i in sorted(zip(angles, lengths))]
  angles = sorted(angles)

  median_size = int(len(lengths) / 25) + 1

  lengths = lengths[-median_size:] + lengths + lengths[:median_size]
  lengths = [sorted(lengths[i:i+median_size*2])[median_size] for i in range(len(lengths) - median_size*2)]

  n = math.acos(lengths[lengths.index(min(lengths))] / 2 ** (1/2)) * 180 / pi
  axis1 = (angles[lengths.index(min(lengths))] + n) % 180
  axis2 = (angles[lengths.index(min(lengths))] - n) % 180

  if axis2 < axis1:
     temp = axis1
     axis1 = axis2
     axis2 = temp

  a = cos(axis1 * pi / 180)
  b = cos(axis2 * pi / 180)
  c = sin(axis1 * pi / 180)
  d = sin(axis2 * pi / 180)

  if plot:
    minxval = loops[0][lengths.index(min(lengths))]
    minyval = loops[1][lengths.index(min(lengths))]

    global figures
    figures += 1
    plt.figure(figures)
    plt.title("Matrix Approx Axis")
    plt.scatter(*loops)
    plt.plot([0, minxval], [0, minyval], color = "red")
    plt.plot([0, a], [0, c], color = "cyan")
    plt.plot([0, b], [0, d], color = "magenta")
    print(a, b)
    print(c, d)
    if not hold_graphs:
      plt.show()

  matrix = [[a, b], [c, d]]

  if eprint or eplot:
     print("Error from best matrix:", errors(matrix, x, y, u, v, plot = eplot))

  return matrix

def predict(matrix, x, y, timestep = 1, amount = 25):
  predictions = [[x, y]]
  matrix = np.array(matrix)

  pos = np.array([x, y])

  for step in range(amount):
    predictions.append(list(np.dot(matrix, pos) * timestep))
    pos = np.array([predictions[-1][0], predictions[-1][1]])

  return np.array(predictions).T

def circlize(x, y, u, v):
  x = np.array(x)
  y = np.array(y)
  u = np.array(u)
  v = np.array(v)

  lengths = (x**2 + y**2)**(1/2)

  return x / lengths, y / lengths, u / lengths, v / lengths

def field_predict(x, y, u, v, a, b, time = 1, amount = 25, momentum = 0):
   x = np.array(x)
   y = np.array(y)
   u = np.array(u)
   v = np.array(v)

   umomentum = 0
   vmomentum = 0

   px = [a]
   py = [b]

   for step in range(amount):
      dist = list((x-px[-1])**2 + (y-py[-1])**2)
      temp = dist.index(min(dist))

      px.append(px[-1] + u[temp] + umomentum)
      py.append(py[-1] + v[temp] + vmomentum)

      umomentum = (px[-1] - px[-2]) * momentum
      vmomentum = (py[-1] - py[-2]) * momentum
  
   return [px, py]

def generate_field(x, y, u, v, scale = 10):
   xbound = max(x) - min(x)
   ybound = max(y) - min(y)

   xlower = min(x)
   ylower = min(y)

   x = np.array(x)
   y = np.array(y)
   u = np.array(u)
   v = np.array(v)

   xbin = xbound / scale
   ybin = ybound / scale

   field = [[], [], [], []]

   for i in range(scale):
      for j in range(scale):
        temp = [k for k in range(len(x)) if xlower + xbin * i < x[k] < xlower + xbin * (i + 1) and ylower + ybin * j < y[k] < ylower + ybin * (j + 1)]

        if temp:
          field[0].append(xlower + xbin * (i+.5))
          field[1].append(ylower + ybin * (j+.5))
          field[2].append(np.mean(u[temp]))
          field[3].append(np.mean(v[temp]))

          x = np.delete(x, temp)
          y = np.delete(y, temp)
          u = np.delete(u, temp)
          v = np.delete(v, temp)

   return field[0], field[1], field[2], field[3]

"""def analysis(x, y, u, v):
  x = np.array(x)
  y = np.array(y)
  u = np.array(u)
  v = np.array(v)

  xflip = []
  uxflip = []
  vxflip = []

  yflip = []
  uyflip = []
  vyflip = []

  lengths = (x**2 + y**2)**(1/2)
  x = x / lengths
  y = y / lengths
  u = u / lengths
  v = v / lengths

  for i in range(len(x)):
    if y[i] < 0:
      xflip.append(-x[i])
      uxflip.append(-u[i])
      vxflip.append(-v[i])
    else:
      xflip.append(x[i])
      uxflip.append(u[i])
      vxflip.append(v[i])

  for i in range(len(x)):
    if x[i] < 0:
      yflip.append(-y[i])
      uyflip.append(-u[i])
      vyflip.append(-v[i])
    else:
      yflip.append(y[i])
      uyflip.append(u[i])
      vyflip.append(v[i])

  plt.scatter(xflip, uxflip, color = "red")
  plt.scatter(xflip, vxflip, color = "green")
  plt.scatter(yflip, uyflip, color = "blue")
  plt.scatter(yflip, vyflip, color = "black")
  plt.show()


def sinregression(sinx, siny):
    sinx = np.array(sinx)
    siny = np.array(siny)
   
    plt.scatter(sinx, siny)

    testy = siny

    diffs = []

    for i in range(len(sinx)):
       diffs.append(np.max(abs(siny + np.roll(testy, i))))

    #using format sin(ax + b)
    b = sinx[diffs.index(min(diffs))]

    for i in range(11):
       b = 1
"""

def find_bars(linx, binlen = 20):
    bigbins = []
    bins = int(360 / binlen) + 1 * bool(360 % binlen)

    for bin in range(bins):
        amt = sum(bin * binlen < x < bin * binlen + binlen for x in linx)
        if amt > len(linx) / bins * 2:
            bigbins.append([bin * binlen, bin * binlen + binlen])

    print("Out of", bins, "bins, big bins on intervals:", bigbins)

    return bigbins

def barzoom(bar, linx, vectorx, vectory, plot = False):
    remove = [i for i in range(len(linx)) if not bar[0] < linx[i] < bar[1]]

    vx = np.delete(np.array(vectorx), remove)
    vy = np.delete(np.array(vectory), remove)

    print("Zooming on bar:", bar, " | ", len(vx), "bars found")
    
    if plot:
        global figures
        figures += 1
        plt.figure(figures)
        plt.title("Subspace Considered")
        plt.plot(vectorx, vectory)
        plt.plot(vx, vy)
        if not hold_graphs:
            plt.show()

    """thetadir = []
    for i in range(len(vx)):
        thetadir.append((math.atan2(vy[i], vx[i]) * 180 / pi) + 360)
    
    theta = abs(max(thetadir) - min(thetadir))
    axisangle = math.atan2(np.mean(vy), np.mean(vx)) * 180 / pi

    n = 3

    vx_far = vx - n * np.mean(vx)
    vy_far = vy - n * math.tan(axisangle * pi / 180) * np.mean(vx)

    farthetadir = []
    for i in range(len(vx)):
        farthetadir.append((math.atan2(vy_far[i], vx_far[i]) * 180 / pi) + 360)

    phi = abs(max(farthetadir) - min(farthetadir))
    extension = n * np.mean(vx) / cos(axisangle * pi / 180)

    center = extension / (1 + math.tan(theta * pi / 180) / math.tan(phi * pi / 180))
    centerx = center * cos(axisangle * pi / 180)
    centery = center * sin(axisangle * pi / 180)"""

    return vx - np.mean(vx), vy - np.mean(vy)

def format_data(cartx, carty):
    x = []
    y = []
    u = []
    v = []

    for i in range(len(cartx) - 1):
        x.append(cartx[i])
        y.append(carty[i])
        u.append(cartx[i + 1] - cartx[i])
        v.append(carty[i + 1] - carty[i])

    return x, y, u, v

def easy_vector_plot(cartx, carty, color = "blue", show = False, stats = False, newfig = True):
    x, y, u, v = format_data(cartx, carty)
    if stats:
      print("# of vectors", len(x))
    plot_vectors(x, y, u, v, color, show, newfig)

def show_all():
    global figures
    print("Now showing all", figures, "figures created")
    plt.show()