import sys
from PyQt5.QtCore import Qt, pyqtSlot

from configs import appInfo

from UI.implement.mainWindows import *
    
from .utils import center


def runUI():
    app = QApplication(sys.argv)

    #화면 전환용 Widget 설정
    widget = QStackedWidget()

    #레이아웃 인스턴스 생성
    firstWindow = FirstWindow(widget)
    dataInputWindow = DataInputWindow(widget)
    analysisWindow = AnalysisWindow(widget)
    resultListWindow = ResultListWindow(widget)
    routeOfConfirmedCaseWindow = RouteOfConfirmedCaseWindow(widget)
    contactorListWindow = ContactorListWindow(widget)

    #Widget 추가
    widget.addWidget(firstWindow)
    widget.addWidget(dataInputWindow)
    widget.addWidget(analysisWindow)
    widget.addWidget(resultListWindow)
    widget.addWidget(routeOfConfirmedCaseWindow)
    widget.addWidget(contactorListWindow)

    #프로그램 화면을 보여주는 코드
    widget.setWindowTitle('CCTV 영상 분석을 통한 코로나 확진자 동선 및 접촉자 판별 시스템')
    widget.resize(1000, 700)
    # 화면을 중앙에 위치시킴
    center(widget)
    widget.show()

    app.aboutToQuit.connect(analysisWindow.onExit)
    
    sys.exit(app.exec_())