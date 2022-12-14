# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Window_FD.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

# 风电预测界面
# 使用QtDesigner工具创建界面生成.ui文件，再通过工具转为.py文件得到，包含风电预测界面的初始化设置

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(919, 760)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_main_title = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_main_title.sizePolicy().hasHeightForWidth())
        self.label_main_title.setSizePolicy(sizePolicy)
        self.label_main_title.setBaseSize(QtCore.QSize(9, 4))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.label_main_title.setFont(font)
        self.label_main_title.setTextFormat(QtCore.Qt.AutoText)
        self.label_main_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_main_title.setObjectName("label_main_title")
        self.verticalLayout_2.addWidget(self.label_main_title, 0, QtCore.Qt.AlignHCenter)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.label_tips = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_tips.sizePolicy().hasHeightForWidth())
        self.label_tips.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.label_tips.setFont(font)
        self.label_tips.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_tips.setWordWrap(False)
        self.label_tips.setObjectName("label_tips")
        self.horizontalLayout_4.addWidget(self.label_tips)
        spacerItem1 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.horizontalLayout_4.setStretch(0, 3)
        self.horizontalLayout_4.setStretch(1, 3)
        self.horizontalLayout_4.setStretch(2, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.tabWidget_select = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget_select.sizePolicy().hasHeightForWidth())
        self.tabWidget_select.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.tabWidget_select.setFont(font)
        self.tabWidget_select.setObjectName("tabWidget_select")
        self.Point_predict = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Point_predict.sizePolicy().hasHeightForWidth())
        self.Point_predict.setSizePolicy(sizePolicy)
        self.Point_predict.setObjectName("Point_predict")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.Point_predict)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_speed70 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_speed70.sizePolicy().hasHeightForWidth())
        self.lineEdit_speed70.setSizePolicy(sizePolicy)
        self.lineEdit_speed70.setObjectName("lineEdit_speed70")
        self.gridLayout.addWidget(self.lineEdit_speed70, 6, 1, 1, 1)
        self.label_predict_value = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_predict_value.sizePolicy().hasHeightForWidth())
        self.label_predict_value.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_predict_value.setFont(font)
        self.label_predict_value.setLineWidth(3)
        self.label_predict_value.setMidLineWidth(0)
        self.label_predict_value.setAlignment(QtCore.Qt.AlignCenter)
        self.label_predict_value.setObjectName("label_predict_value")
        self.gridLayout.addWidget(self.label_predict_value, 7, 4, 1, 1)
        self.lineEdit_humidity = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_humidity.sizePolicy().hasHeightForWidth())
        self.lineEdit_humidity.setSizePolicy(sizePolicy)
        self.lineEdit_humidity.setObjectName("lineEdit_humidity")
        self.gridLayout.addWidget(self.lineEdit_humidity, 6, 4, 1, 1)
        self.label_speed70 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_speed70.sizePolicy().hasHeightForWidth())
        self.label_speed70.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_speed70.setFont(font)
        self.label_speed70.setLineWidth(3)
        self.label_speed70.setMidLineWidth(0)
        self.label_speed70.setAlignment(QtCore.Qt.AlignCenter)
        self.label_speed70.setObjectName("label_speed70")
        self.gridLayout.addWidget(self.label_speed70, 6, 0, 1, 1)
        self.label_humidity = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_humidity.sizePolicy().hasHeightForWidth())
        self.label_humidity.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_humidity.setFont(font)
        self.label_humidity.setLineWidth(3)
        self.label_humidity.setMidLineWidth(0)
        self.label_humidity.setAlignment(QtCore.Qt.AlignCenter)
        self.label_humidity.setObjectName("label_humidity")
        self.gridLayout.addWidget(self.label_humidity, 6, 3, 1, 1)
        self.label_predict_power = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_predict_power.sizePolicy().hasHeightForWidth())
        self.label_predict_power.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_predict_power.setFont(font)
        self.label_predict_power.setLineWidth(3)
        self.label_predict_power.setMidLineWidth(0)
        self.label_predict_power.setAlignment(QtCore.Qt.AlignCenter)
        self.label_predict_power.setObjectName("label_predict_power")
        self.gridLayout.addWidget(self.label_predict_power, 7, 3, 1, 1)
        self.lineEdit_direction50 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_direction50.sizePolicy().hasHeightForWidth())
        self.lineEdit_direction50.setSizePolicy(sizePolicy)
        self.lineEdit_direction50.setObjectName("lineEdit_direction50")
        self.gridLayout.addWidget(self.lineEdit_direction50, 5, 1, 1, 1)
        self.lineEdit_direction70 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_direction70.sizePolicy().hasHeightForWidth())
        self.lineEdit_direction70.setSizePolicy(sizePolicy)
        self.lineEdit_direction70.setObjectName("lineEdit_direction70")
        self.gridLayout.addWidget(self.lineEdit_direction70, 7, 1, 1, 1)
        self.lineEdit_pressure = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_pressure.sizePolicy().hasHeightForWidth())
        self.lineEdit_pressure.setSizePolicy(sizePolicy)
        self.lineEdit_pressure.setObjectName("lineEdit_pressure")
        self.gridLayout.addWidget(self.lineEdit_pressure, 5, 4, 1, 1)
        self.label_pressure = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_pressure.sizePolicy().hasHeightForWidth())
        self.label_pressure.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_pressure.setFont(font)
        self.label_pressure.setLineWidth(3)
        self.label_pressure.setMidLineWidth(0)
        self.label_pressure.setAlignment(QtCore.Qt.AlignCenter)
        self.label_pressure.setObjectName("label_pressure")
        self.gridLayout.addWidget(self.label_pressure, 5, 3, 1, 1)
        self.label_direction70 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_direction70.sizePolicy().hasHeightForWidth())
        self.label_direction70.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_direction70.setFont(font)
        self.label_direction70.setLineWidth(3)
        self.label_direction70.setMidLineWidth(0)
        self.label_direction70.setAlignment(QtCore.Qt.AlignCenter)
        self.label_direction70.setObjectName("label_direction70")
        self.gridLayout.addWidget(self.label_direction70, 7, 0, 1, 1)
        self.label_direction50 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_direction50.sizePolicy().hasHeightForWidth())
        self.label_direction50.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_direction50.setFont(font)
        self.label_direction50.setLineWidth(3)
        self.label_direction50.setMidLineWidth(0)
        self.label_direction50.setAlignment(QtCore.Qt.AlignCenter)
        self.label_direction50.setObjectName("label_direction50")
        self.gridLayout.addWidget(self.label_direction50, 5, 0, 1, 1)
        self.lineEdit_direction30 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_direction30.sizePolicy().hasHeightForWidth())
        self.lineEdit_direction30.setSizePolicy(sizePolicy)
        self.lineEdit_direction30.setObjectName("lineEdit_direction30")
        self.gridLayout.addWidget(self.lineEdit_direction30, 3, 1, 1, 1)
        self.lineEdit_speed10 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_speed10.sizePolicy().hasHeightForWidth())
        self.lineEdit_speed10.setSizePolicy(sizePolicy)
        self.lineEdit_speed10.setText("")
        self.lineEdit_speed10.setObjectName("lineEdit_speed10")
        self.gridLayout.addWidget(self.lineEdit_speed10, 0, 1, 1, 1)
        self.label_direction100 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_direction100.sizePolicy().hasHeightForWidth())
        self.label_direction100.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_direction100.setFont(font)
        self.label_direction100.setLineWidth(3)
        self.label_direction100.setMidLineWidth(0)
        self.label_direction100.setAlignment(QtCore.Qt.AlignCenter)
        self.label_direction100.setObjectName("label_direction100")
        self.gridLayout.addWidget(self.label_direction100, 3, 3, 1, 1)
        self.lineEdit_direction90 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_direction90.sizePolicy().hasHeightForWidth())
        self.lineEdit_direction90.setSizePolicy(sizePolicy)
        self.lineEdit_direction90.setObjectName("lineEdit_direction90")
        self.gridLayout.addWidget(self.lineEdit_direction90, 1, 4, 1, 1)
        self.lineEdit_speed100 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_speed100.sizePolicy().hasHeightForWidth())
        self.lineEdit_speed100.setSizePolicy(sizePolicy)
        self.lineEdit_speed100.setObjectName("lineEdit_speed100")
        self.gridLayout.addWidget(self.lineEdit_speed100, 2, 4, 1, 1)
        self.label_direction10 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_direction10.sizePolicy().hasHeightForWidth())
        self.label_direction10.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_direction10.setFont(font)
        self.label_direction10.setLineWidth(3)
        self.label_direction10.setMidLineWidth(0)
        self.label_direction10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_direction10.setObjectName("label_direction10")
        self.gridLayout.addWidget(self.label_direction10, 1, 0, 1, 1)
        self.lineEdit_direction100 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_direction100.sizePolicy().hasHeightForWidth())
        self.lineEdit_direction100.setSizePolicy(sizePolicy)
        self.lineEdit_direction100.setObjectName("lineEdit_direction100")
        self.gridLayout.addWidget(self.lineEdit_direction100, 3, 4, 1, 1)
        self.label_speed100 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_speed100.sizePolicy().hasHeightForWidth())
        self.label_speed100.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_speed100.setFont(font)
        self.label_speed100.setLineWidth(3)
        self.label_speed100.setMidLineWidth(0)
        self.label_speed100.setAlignment(QtCore.Qt.AlignCenter)
        self.label_speed100.setObjectName("label_speed100")
        self.gridLayout.addWidget(self.label_speed100, 2, 3, 1, 1)
        self.label_direction30 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_direction30.sizePolicy().hasHeightForWidth())
        self.label_direction30.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_direction30.setFont(font)
        self.label_direction30.setLineWidth(3)
        self.label_direction30.setMidLineWidth(0)
        self.label_direction30.setAlignment(QtCore.Qt.AlignCenter)
        self.label_direction30.setObjectName("label_direction30")
        self.gridLayout.addWidget(self.label_direction30, 3, 0, 1, 1)
        self.lineEdit_speed90 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_speed90.sizePolicy().hasHeightForWidth())
        self.lineEdit_speed90.setSizePolicy(sizePolicy)
        self.lineEdit_speed90.setObjectName("lineEdit_speed90")
        self.gridLayout.addWidget(self.lineEdit_speed90, 0, 4, 1, 1)
        self.lineEdit_direction10 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_direction10.sizePolicy().hasHeightForWidth())
        self.lineEdit_direction10.setSizePolicy(sizePolicy)
        self.lineEdit_direction10.setObjectName("lineEdit_direction10")
        self.gridLayout.addWidget(self.lineEdit_direction10, 1, 1, 1, 1)
        self.label_speed10 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_speed10.sizePolicy().hasHeightForWidth())
        self.label_speed10.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_speed10.setFont(font)
        self.label_speed10.setLineWidth(3)
        self.label_speed10.setMidLineWidth(0)
        self.label_speed10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_speed10.setObjectName("label_speed10")
        self.gridLayout.addWidget(self.label_speed10, 0, 0, 1, 1)
        self.label_temper = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_temper.sizePolicy().hasHeightForWidth())
        self.label_temper.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_temper.setFont(font)
        self.label_temper.setLineWidth(3)
        self.label_temper.setMidLineWidth(0)
        self.label_temper.setAlignment(QtCore.Qt.AlignCenter)
        self.label_temper.setObjectName("label_temper")
        self.gridLayout.addWidget(self.label_temper, 4, 3, 1, 1)
        self.lineEdit_temper = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_temper.sizePolicy().hasHeightForWidth())
        self.lineEdit_temper.setSizePolicy(sizePolicy)
        self.lineEdit_temper.setObjectName("lineEdit_temper")
        self.gridLayout.addWidget(self.lineEdit_temper, 4, 4, 1, 1)
        self.label_speed50 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_speed50.sizePolicy().hasHeightForWidth())
        self.label_speed50.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_speed50.setFont(font)
        self.label_speed50.setLineWidth(3)
        self.label_speed50.setMidLineWidth(0)
        self.label_speed50.setAlignment(QtCore.Qt.AlignCenter)
        self.label_speed50.setObjectName("label_speed50")
        self.gridLayout.addWidget(self.label_speed50, 4, 0, 1, 1)
        self.label_speed30 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_speed30.sizePolicy().hasHeightForWidth())
        self.label_speed30.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_speed30.setFont(font)
        self.label_speed30.setLineWidth(3)
        self.label_speed30.setMidLineWidth(0)
        self.label_speed30.setAlignment(QtCore.Qt.AlignCenter)
        self.label_speed30.setObjectName("label_speed30")
        self.gridLayout.addWidget(self.label_speed30, 2, 0, 1, 1)
        self.label_direction90 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_direction90.sizePolicy().hasHeightForWidth())
        self.label_direction90.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_direction90.setFont(font)
        self.label_direction90.setLineWidth(3)
        self.label_direction90.setMidLineWidth(0)
        self.label_direction90.setAlignment(QtCore.Qt.AlignCenter)
        self.label_direction90.setObjectName("label_direction90")
        self.gridLayout.addWidget(self.label_direction90, 1, 3, 1, 1)
        self.lineEdit_speed30 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_speed30.sizePolicy().hasHeightForWidth())
        self.lineEdit_speed30.setSizePolicy(sizePolicy)
        self.lineEdit_speed30.setObjectName("lineEdit_speed30")
        self.gridLayout.addWidget(self.lineEdit_speed30, 2, 1, 1, 1)
        self.lineEdit_speed50 = QtWidgets.QLineEdit(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_speed50.sizePolicy().hasHeightForWidth())
        self.lineEdit_speed50.setSizePolicy(sizePolicy)
        self.lineEdit_speed50.setObjectName("lineEdit_speed50")
        self.gridLayout.addWidget(self.lineEdit_speed50, 4, 1, 1, 1)
        self.label_speed90 = QtWidgets.QLabel(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_speed90.sizePolicy().hasHeightForWidth())
        self.label_speed90.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_speed90.setFont(font)
        self.label_speed90.setLineWidth(3)
        self.label_speed90.setMidLineWidth(0)
        self.label_speed90.setAlignment(QtCore.Qt.AlignCenter)
        self.label_speed90.setObjectName("label_speed90")
        self.gridLayout.addWidget(self.label_speed90, 0, 3, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 3, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.pushButton_predict = QtWidgets.QPushButton(self.Point_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_predict.sizePolicy().hasHeightForWidth())
        self.pushButton_predict.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_predict.setFont(font)
        self.pushButton_predict.setObjectName("pushButton_predict")
        self.verticalLayout.addWidget(self.pushButton_predict, 0, QtCore.Qt.AlignHCenter)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)
        self.tabWidget_select.addTab(self.Point_predict, "")
        self.File_predict = QtWidgets.QWidget()
        self.File_predict.setObjectName("File_predict")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.File_predict)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem6)
        self.pushButton_forFileDialog = QtWidgets.QPushButton(self.File_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_forFileDialog.sizePolicy().hasHeightForWidth())
        self.pushButton_forFileDialog.setSizePolicy(sizePolicy)
        self.pushButton_forFileDialog.setSizeIncrement(QtCore.QSize(0, 0))
        self.pushButton_forFileDialog.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_forFileDialog.setFont(font)
        self.pushButton_forFileDialog.setObjectName("pushButton_forFileDialog")
        self.verticalLayout_4.addWidget(self.pushButton_forFileDialog, 0, QtCore.Qt.AlignHCenter)
        spacerItem7 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_4.addItem(spacerItem7)
        self.label_file_ok = QtWidgets.QLabel(self.File_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_file_ok.sizePolicy().hasHeightForWidth())
        self.label_file_ok.setSizePolicy(sizePolicy)
        self.label_file_ok.setWordWrap(True)
        self.label_file_ok.setObjectName("label_file_ok")
        self.verticalLayout_4.addWidget(self.label_file_ok)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem8)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        spacerItem9 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem9)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_chart = QtWidgets.QLabel(self.File_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(50)
        sizePolicy.setHeightForWidth(self.label_chart.sizePolicy().hasHeightForWidth())
        self.label_chart.setSizePolicy(sizePolicy)
        self.label_chart.setText("")
        self.label_chart.setObjectName("label_chart")
        self.verticalLayout_3.addWidget(self.label_chart)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem10)
        self.pushButton_save_chart = QtWidgets.QPushButton(self.File_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_save_chart.sizePolicy().hasHeightForWidth())
        self.pushButton_save_chart.setSizePolicy(sizePolicy)
        self.pushButton_save_chart.setObjectName("pushButton_save_chart")
        self.horizontalLayout.addWidget(self.pushButton_save_chart)
        self.pushButton_w_tofile = QtWidgets.QPushButton(self.File_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_w_tofile.sizePolicy().hasHeightForWidth())
        self.pushButton_w_tofile.setSizePolicy(sizePolicy)
        self.pushButton_w_tofile.setObjectName("pushButton_w_tofile")
        self.horizontalLayout.addWidget(self.pushButton_w_tofile)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem11)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 2)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        spacerItem12 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_3.addItem(spacerItem12)
        self.verticalLayout_3.setStretch(0, 8)
        self.verticalLayout_3.setStretch(1, 1)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.pushButton_predict_file = QtWidgets.QPushButton(self.File_predict)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_predict_file.sizePolicy().hasHeightForWidth())
        self.pushButton_predict_file.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_predict_file.setFont(font)
        self.pushButton_predict_file.setObjectName("pushButton_predict_file")
        self.verticalLayout_5.addWidget(self.pushButton_predict_file, 0, QtCore.Qt.AlignHCenter)
        spacerItem13 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout_5.addItem(spacerItem13)
        self.tabWidget_select.addTab(self.File_predict, "")
        self.verticalLayout_2.addWidget(self.tabWidget_select)
        self.verticalLayout_2.setStretch(0, 2)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 10)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 919, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget_select.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "新能源预测软件(风电)"))
        self.label_main_title.setText(_translate("MainWindow", "风力发电预测"))
        self.label_tips.setText(_translate("MainWindow", "请输入数据进行点预测或选择规范的文件进行预测"))
        self.lineEdit_speed70.setPlaceholderText(_translate("MainWindow", "0-100"))
        self.label_predict_value.setText(_translate("MainWindow", "MW"))
        self.lineEdit_humidity.setPlaceholderText(_translate("MainWindow", "0-100"))
        self.label_speed70.setText(_translate("MainWindow", "Speed70(m/s)"))
        self.label_humidity.setText(_translate("MainWindow", "Humidity(%)"))
        self.label_predict_power.setText(_translate("MainWindow", "Power(predict)"))
        self.lineEdit_direction50.setPlaceholderText(_translate("MainWindow", "0-360"))
        self.lineEdit_direction70.setPlaceholderText(_translate("MainWindow", "0-360"))
        self.lineEdit_pressure.setPlaceholderText(_translate("MainWindow", "0-5000"))
        self.label_pressure.setText(_translate("MainWindow", "Pressure(hPa)"))
        self.label_direction70.setText(_translate("MainWindow", "Direction70(度)"))
        self.label_direction50.setText(_translate("MainWindow", "Direction50(度)"))
        self.lineEdit_direction30.setPlaceholderText(_translate("MainWindow", "0-360"))
        self.lineEdit_speed10.setPlaceholderText(_translate("MainWindow", "0-100"))
        self.label_direction100.setText(_translate("MainWindow", "Direction100(度)"))
        self.lineEdit_direction90.setPlaceholderText(_translate("MainWindow", "0-360"))
        self.lineEdit_speed100.setPlaceholderText(_translate("MainWindow", "0-100"))
        self.label_direction10.setText(_translate("MainWindow", "Direction10(度)"))
        self.lineEdit_direction100.setPlaceholderText(_translate("MainWindow", "0-360"))
        self.label_speed100.setText(_translate("MainWindow", "Speed100(m/s)"))
        self.label_direction30.setText(_translate("MainWindow", "Direction30(度)"))
        self.lineEdit_speed90.setPlaceholderText(_translate("MainWindow", "0-100"))
        self.lineEdit_direction10.setPlaceholderText(_translate("MainWindow", "0-360"))
        self.label_speed10.setText(_translate("MainWindow", "Speed10(m/s)"))
        self.label_temper.setText(_translate("MainWindow", "Temper(K)"))
        self.lineEdit_temper.setPlaceholderText(_translate("MainWindow", "-60-100"))
        self.label_speed50.setText(_translate("MainWindow", "Speed50(m/s)"))
        self.label_speed30.setText(_translate("MainWindow", "Speed30(m/s)"))
        self.label_direction90.setText(_translate("MainWindow", "Direction90(度)"))
        self.lineEdit_speed30.setPlaceholderText(_translate("MainWindow", "0-100"))
        self.lineEdit_speed50.setPlaceholderText(_translate("MainWindow", "0-100"))
        self.label_speed90.setText(_translate("MainWindow", "Speed90(m/s)"))
        self.pushButton_predict.setText(_translate("MainWindow", "开始预测"))
        self.tabWidget_select.setTabText(self.tabWidget_select.indexOf(self.Point_predict), _translate("MainWindow", "点预测"))
        self.pushButton_forFileDialog.setText(_translate("MainWindow", "选择文件"))
        self.label_file_ok.setText(_translate("MainWindow", "请保证文件格式与示例文件相同，即前16列依次为......"))
        self.pushButton_save_chart.setText(_translate("MainWindow", "保存图表"))
        self.pushButton_w_tofile.setText(_translate("MainWindow", "将结果写入文件"))
        self.pushButton_predict_file.setText(_translate("MainWindow", "开始预测"))
        self.tabWidget_select.setTabText(self.tabWidget_select.indexOf(self.File_predict), _translate("MainWindow", "文件预测"))
