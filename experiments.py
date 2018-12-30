from layout import getLayout
from pacman import *
from submission import *
from ghostAgents import *
from textDisplay import *

# Notification Balloon imports
# from win32api import *
# from win32gui import *
# import win32con
# import sys, os
# import struct
# import time


# class WindowsBalloonTip:
#     def __init__(self, title, msg):
#         message_map = {
#             win32con.WM_DESTROY: self.OnDestroy,
#         }
#         # Register the Window class.
#         wc = WNDCLASS()
#         hinst = wc.hInstance = GetModuleHandle(None)
#         wc.lpszClassName = "PythonTaskbar"
#         wc.lpfnWndProc = message_map  # could also specify a wndproc.
#         classAtom = RegisterClass(wc)
#         # Create the Window.
#         style = win32con.WS_OVERLAPPED | win32con.WS_SYSMENU
#         self.hwnd = CreateWindow(classAtom, "Taskbar", style, \
#                                  0, 0, win32con.CW_USEDEFAULT, win32con.CW_USEDEFAULT, \
#                                  0, 0, hinst, None)
#         UpdateWindow(self.hwnd)
#         iconPathName = os.path.abspath(os.path.join(sys.path[0], "balloontip.ico"))
#         icon_flags = win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
#         try:
#             hicon = LoadImage(hinst, iconPathName, \
#                               win32con.IMAGE_ICON, 0, 0, icon_flags)
#         except:
#             hicon = LoadIcon(0, win32con.IDI_APPLICATION)
#         flags = NIF_ICON | NIF_MESSAGE | NIF_TIP
#         nid = (self.hwnd, 0, flags, win32con.WM_USER + 20, hicon, "tooltip")
#         Shell_NotifyIcon(NIM_ADD, nid)
#         Shell_NotifyIcon(NIM_MODIFY, \
#                          (self.hwnd, 0, NIF_INFO, win32con.WM_USER + 20, \
#                           hicon, "Balloon  tooltip", msg, 200, title))
#         # self.show_balloon(title, msg)
#         time.sleep(10)
#         DestroyWindow(self.hwnd)
#
#     def OnDestroy(self, hwnd, msg, wparam, lparam):
#         nid = (self.hwnd, 0)
#         Shell_NotifyIcon(NIM_DELETE, nid)
#         PostQuitMessage(0)  # Terminate the app.


# def balloon_tip(title, msg):
#     w = WindowsBalloonTip(title, msg)


players = [OriginalReflexAgent, ReflexAgent, MinimaxAgent, AlphaBetaAgent, RandomExpectimaxAgent]
# players = [RandomExpectimaxAgent]
depths = [2, 3, 4]
layouts = ['capsuleClassic', 'contestClassic', 'mediumClassic',
           'minimaxClassic', 'openClassic', 'originalClassic',
           'smallClassic', 'testClassic', 'trappedClassic', 'trickyClassic']
ghosts = [RandomGhost(1), RandomGhost(2)]


def run_game(player, layout_name, file_ptr, depth=1):
    layout = getLayout(layout_name)
    if depth > 1:
        player.depth = depth

    games = runGames(layout, player, ghosts, NullGraphics(), 7, False, 0, False, 30)
    scores = [game.state.getScore() for game in games]
    times = [game.my_avg_time for game in games]
    avg_score = sum(scores) / float(len(scores))
    avg_time = sum(times) / float(len(times))
    line = (player.__class__.__name__ + ',' +
            str(depth) + ',' +
            layout_name + ',' +
            '%.2f' % avg_score + ',' +
            '%.2f' % (avg_time * 1e6) + 'E-06\n')
    file_ptr.write(line)
    return


if __name__ == '__main__':
    # balloon_tip('AI HW2', 'Experiments Started');
    base = time.time()
    with open('experiments.csv', 'w+') as file_ptr:
        for layout in layouts:
            for player in players:
                if player in [OriginalReflexAgent, ReflexAgent]:
                    print(layout, player)
                    run_game(player(), layout, file_ptr)
                else:
                    for d in depths:
                        # print(layout, player, f' depth={d}')
                        run_game(player(), layout, file_ptr, d)
            file_ptr.write('\n')  # TODO: remove this before submitting
    file_ptr.close()
    print(f'experiments time: {(time.time() - base) / 60} min')
    # balloon_tip('AI HW2', 'Experiments Finished!');
