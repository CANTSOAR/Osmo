import phasefield as pf
import stockstuff as ss
import numpy as np

def normalize(array):
    array = array / np.max(abs(array))
    return array

def macd_rsi_full_analysis(stock, interval = "1d", btrange = 12, btamount = 200, zoom_ang = 40, pred_range = 25, plot = False):
    pf.hold_graphs = True

    prices, _, _, _ = ss.create_data(stock, btrange, interval)

    print("Available Data Points:", len(prices))

    rsis = ss.calc_rsis(prices)
    _, _, macds = ss.calc_macd(prices)

    rsis = ss.calc_ema(rsis)[-btamount:]
    macds = ss.calc_ema(macds)[-btamount:]

    rsis = rsis - np.mean(rsis)

    rsis = normalize(rsis)
    macds = normalize(macds)

    x, y, u, v = pf.format_data(rsis, macds)
    a, b, c, d = pf.generate_field(x, y, u, v)
    matrix = pf.best_matrix(*pf.circlize(a, b, c, d), plot = plot, eplot = plot, eprint = True)
    if plot:
        print(matrix)
        pf.easy_vector_plot(rsis, macds, color = pf.color_gradient(len(rsis)))
        pf.easy_vector_plot(*pf.predict(matrix, rsis[-1], macds[-1], 1, pred_range), color = pf.color_gradient(pred_range))
        pf.plot_vectors(a, b, c, d, color = "black")
        pf.easy_vector_plot(*pf.field_predict(a, b, c, d, rsis[-1], macds[-1], 1, pred_range, momentum = .5), color = pf.color_gradient(pred_range), newfig = False)

    linx, _ = pf.linearize(x, y, u, v)
    bars = pf.find_bars(linx, zoom_ang)

    for bar in bars:
        barx, bary = pf.barzoom(bar, linx, x, y)
        x1, y1, u1, v1 = pf.format_data(barx, bary)
        a1, b1, c1, d1 = pf.generate_field(x1, y1, u1, v1)
        matrix = pf.best_matrix(*pf.circlize(a1, b1, c1, d1), plot = plot, eplot = plot, eprint = True)
        if plot:
            print(matrix)
            pf.easy_vector_plot(*pf.predict(matrix, rsis[-1], macds[-1], 1, pred_range), color = pf.color_gradient(pred_range))
            pf.plot_vectors(a1, b1, c1, d1, color = "black")
            pf.easy_vector_plot(*pf.field_predict(a1, b1, c1, d1, rsis[-1], macds[-1], 1, pred_range, momentum = .5), color = pf.color_gradient(pred_range), newfig = False)

        barlinx, _ = pf.linearize(x1, y1, u1, v1)
        bars2 = pf.find_bars(barlinx, zoom_ang)

        for bar2 in bars2:
            barx, bary = pf.barzoom(bar2, barlinx, x1, y1)
            x2, y2, u2, v2 = pf.format_data(barx, bary)
            a2, b2, c2, d2 = pf.generate_field(x2, y2, u2, v2)
            matrix = pf.best_matrix(*pf.circlize(a2, b2, c2, d2), plot = plot, eplot = plot, eprint = True)
            if plot:
                print(matrix)
                pf.easy_vector_plot(*pf.predict(matrix, rsis[-1], macds[-1], 1, pred_range), color = pf.color_gradient(pred_range))
                pf.plot_vectors(a2, b2, c2, d2, color = "black")
                pf.easy_vector_plot(*pf.field_predict(a2, b2, c2, d2, rsis[-1], macds[-1], 1, pred_range, momentum = .5), color = pf.color_gradient(pred_range), newfig = False)
            

    pf.show_all()


stocks = ["TQQQ", "SQQQ"]
ratios = [.5, .5]
ss.sharpe(stocks, ratios, show = True, btrange = 24)
ss.plot_simple_holdings(stocks, ratios, btrange = 24)