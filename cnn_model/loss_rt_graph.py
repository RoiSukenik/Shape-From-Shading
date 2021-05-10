# """Thermal REALTIME graph - 30.3.21"""
# """
# Implantation:
#
# 1)
# AT THE TEST FILE PY:
# import from this file the func 'start_graph_thread' to the py file which your test func is located:
# from 'path_to_this_file'.loss_rt_graph import start_graph_thread
#
# then at the test function insert at the beginning:
# start_graph_thread()
#
# 2)
# AT THE THERMAL.PY FILE:
# import this file:
# from 'path_to_this_file' import loss_rt_graph
#
# At the save_telemetry function insert after the row 'data = get_telemetries..' this line:
# loss_rt_graph.graph_data = data
# """
# import sys
# import threading
# from datetime import datetime, timedelta
#
# from PyQt5 import QtCore, QtWidgets, QtGui
# from PyQt5.QtWidgets import QLabel
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.figure import Figure
#
# graph_data = []  # global variable for storing the sample thermal data
# TIME_FRAME = 3600  # time frame for the graph in seconds
#
# """Class for handling Thermal sample data"""
# class Thermal_Sample():
#     def __init__(self, data):
#         self.date = data[0]
#         self.loss_avg = data[1]
#         self.loss = data[2]
#         self.epoch = data[3]
#
#
#     def __eq__(self, other):
#         if not isinstance(other, Thermal_Sample):
#             # don't attempt to compare against unrelated types
#             return NotImplemented
#         return self.date == other.date and self.loss_avg == other.loss_avg and self.loss == other.loss
#
#
# """Gui for showing the graph"""
# class Window(QtWidgets.QMainWindow):
#     def __init__(self):
#         QtWidgets.QMainWindow.__init__(self)
#         self.widget = QtWidgets.QWidget()
#         self.setCentralWidget(self.widget)
#         self.widget.setLayout(QtWidgets.QVBoxLayout())
#         self.widget.layout().setContentsMargins(0, 0, 0, 0)
#         self.widget.layout().setSpacing(0)
#         self.startTime = datetime.now()
#
#         self.fig = Figure(figsize=(12, 12), dpi=100)
#
#         self.axes = self.fig.add_subplot(111)
#         self.axes.grid()
#         self.axes.set_title('Loss Live Graph')
#
#         self.control_label = QLabel("Time Elapsed: 00:00:00    Loss: 0    Loss AVG: 0", self)
#
#         self.control_label.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Black))
#         self.control_label.setFixedWidth(900)
#         self.control_label.move(200, 40)
#
#         self.xdata = [0]
#         self.ydata = [0]
#         self.entry_limit = TIME_FRAME
#
#         self.line, = self.axes.plot([], [], 'r', lw=3, label='Loss AVG')
#
#         self.axes2 = self.axes.twinx()
#         self.axes2.set_ylabel("LOSS")
#         self.axes.set_ylabel("LOSS - AVG")
#         self.axes.set_xlabel("Time Passed(Sec)")
#
#         self.y2data = [0]
#         self.line2, = self.axes2.plot([], [], 'b', label='Loss')
#         self.plt_colors = ['g', 'r', 'c', 'm', 'y', 'k']
#
#         # joined legend for 3 plots
#         handles, labels = [], []
#         for ax in self.fig.axes:
#             for h, l in zip(*ax.get_legend_handles_labels()):
#                 handles.append(h)
#                 labels.append(l)
#         self.axes.legend(handles, labels, loc=4, fontsize=8)
#
#         self.last_sample = None
#         self.canvas = FigureCanvas(self.fig)
#         self.canvas.draw()
#
#         self.nav = NavigationToolbar(self.canvas, self.widget)
#         self.widget.layout().addWidget(self.nav)
#         self.widget.layout().addWidget(self.canvas)
#
#         self.show()
#
#         self.ctimer = QtCore.QTimer()
#         self.ctimer.timeout.connect(self.update)
#         self.ctimer.start(150)
#
#     def update(self):
#
#         if graph_data:
#             new_sample = Thermal_Sample(graph_data)
#             if self.last_sample:
#                 if new_sample != self.last_sample:
#                     self.update_figure_with_new_value(new_sample)
#             else: # first time
#                 self.startTime = datetime.strptime(new_sample.date, '%Y/%m/%d_%H:%M:%S')
#                 self.last_sample = new_sample
#                 self.update_figure_with_new_value(new_sample)
#             time_passed = str(timedelta(seconds=round((datetime.now() - self.startTime).total_seconds())))
#             self.control_label.setText(
#                 f"Time Elapsed: {time_passed}    LOSS: {round(self.last_sample.loss, 2)}     Loss AVG: {round(self.last_sample.loss_avg, 2)}")
#
#     def update_figure_with_new_value(self, new_sample):
#         sample_time = datetime.strptime(new_sample.date, '%Y/%m/%d_%H:%M:%S')
#
#         xval = (sample_time - self.startTime).total_seconds()
#
#         self.xdata.append(xval)
#         self.ydata.append(new_sample.loss_avg)
#         self.y2data.append(new_sample.loss)
#
#         if len(self.xdata) > self.entry_limit:
#             self.xdata.pop(0)
#             self.ydata.pop(0)
#             self.y2data.pop(0)
#
#         self.line.set_data(self.xdata, self.ydata)
#         self.axes.relim()
#         self.axes.autoscale_view()
#
#         if self.last_sample.epoch != new_sample.epoch:
#             new_c = self.plt_colors.pop(0)
#             self.plt_colors.append(new_c)
#             self.axes.axvline(x=xval, label='epoch = {}'.format(new_sample.epoch), linestyle="dashed",color=new_c)
#
#             handles, labels = [], []
#             for ax in self.fig.axes:
#                 for h, l in zip(*ax.get_legend_handles_labels()):
#                     handles.append(h)
#                     labels.append(l)
#             self.axes.legend(handles, labels, loc=4, fontsize=8)
#
#         self.last_sample = new_sample
#
#         self.line2.set_data(self.xdata, self.y2data)
#         self.axes2.relim()
#         self.axes2.autoscale_view()
#
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()
#
#
# """Graph Class that can be called as thread"""
# class RT_Graph(threading.Thread):
#     def __init__(self, threadID, name):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#
#     def run(self):
#         qapp = QtWidgets.QApplication(sys.argv)
#         window_instance = Window()
#         exit(qapp.exec_())
#
#
# def start_graph_thread():
#     threads = []
#     # Create new threads
#     graph_thread = RT_Graph(1, "RT_Graph")
#     # Start new Threads
#     graph_thread.start()
#     # Add threads to thread list
#     threads.append(graph_thread)
