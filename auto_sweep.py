import pdb
import time
import cv2
import torch
from torchvision import transforms
import numpy as np
import pyautogui
import pygetwindow as gw
from tqdm import tqdm
from minist_net import MinistNet

def capture_window(window_title):
    # 获取所有窗口
    all_windows = gw.getAllTitles()
    # 查找包含指定关键词的窗口标题（例如包含 Edge 的标题）
    browser_window_titles = [title for title in all_windows if window_title in title]
    chosen_browser_title = browser_window_titles[0]
    # 获取选定浏览器窗口对象
    window = gw.getWindowsWithTitle(chosen_browser_title)[0]
    # 将窗口置顶（可选，如果窗口未被遮挡则不需要）
    window.activate()
    time.sleep(0.5)
    # 获取窗口在屏幕上的位置和大小
    x, y, width, height = window.left, window.top, window.width, window.height
    return x, y, width, height

def is_point_inside_rect(px, py, rect):
    """判断一个点 (px, py) 是否在矩形内"""
    # 获取矩形的最小和最大 x、y 坐标
    min_x, min_y = rect[0]
    max_x, max_y = rect[1]
    
    return min_x - 1 <= px <= max_x + 1 and min_y - 1 <= py <= max_y + 1


def is_rectangle_inside(rect1, rect2):
    """
    判断矩形 rect1 是否完全在矩形 rect2 内。
    参数:
    rect1, rect2: 这两个矩形的格式为 numpy 4x2 数组， 每一行表示矩形的一个角点坐标 (x, y)。
    返回:
    True 如果 rect1 完全在 rect2 内部，否则返回 False。
    """
    # 检查 rect1 的四个角是否都在 rect2 内
    for (px, py) in rect1:
        if not is_point_inside_rect(px, py, rect2):
            return False
    return True


def get_board_pos(image):
    # 转换为灰度图  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # 显示处理后的图像  
    # cv2.imshow('gray Image', gray)  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()
    # 应用阈值处理  
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # # 显示处理后的图像  
    # cv2.imshow('Thresholded Image', thresh)  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()

    # 查找轮廓  
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
    rects = []
    # 遍历轮廓  
    for cnt in contours:  
        # 近似轮廓为多边形  
        epsilon = 0.1 * cv2.arcLength(cnt, True)  
        approx = cv2.approxPolyDP(cnt, epsilon, True) 
        # 如果多边形有4个顶点，则可能是矩形  
        if len(approx) == 4:  
            # 绘制轮廓  
            rect = approx.reshape(4, 2)
            lt, br = rect.min(axis=0), rect.max(axis=0)
            area = (br[0] - lt[0]) * (br[1] - lt[1])
            rect = np.array([lt, br])
            rects.append((rect, area))
    rects.sort(key=lambda x:x[1])
    rects = rects[::-1]
    # pdb.set_trace()
    bbox = rects[0][0]
    grids = []
    for rect in rects[1:]:
        if is_rectangle_inside(rect[0], bbox):
            grids.append(rect[0])
    grids.sort(key=lambda x: (x[0][0], x[0][1]))
    # pdb.set_trace()
    num_row = np.sum([abs(g[0][0] - grids[0][0][0]) < 5 for g in grids[:100]])
    grids.sort(key=lambda x: (x[0][1], x[0][0]))
    num_col = np.sum([abs(g[0][1] - grids[0][0][1]) < 5 for g in grids[:100]])
    grid_wh = np.round(np.mean((bbox[1] - bbox[0]) / np.array([num_col, num_row]))).astype(np.int32)
    boards_center = np.zeros((num_row, num_col, 2), dtype=np.float32)
    xs = np.linspace(bbox[0][0] + grid_wh / 2, bbox[1][0] - grid_wh / 2, num_col)
    ys = np.linspace(bbox[0][1] + grid_wh / 2, bbox[1][1] - grid_wh / 2, num_row)
    xs, ys = np.meshgrid(xs, ys)
    boards_center = np.stack([xs, ys], axis=-1)
    boards_center = np.round(boards_center).astype(np.int32)
    tl = boards_center - grid_wh / 2
    br = boards_center + grid_wh / 2
    boards_pos = np.stack([tl, br], axis=-2).astype(np.int32)
    for grid in boards_pos.reshape(-1, 2, 2):
        cv2.rectangle(image, grid[0], grid[1], (0, 255, 0), 2)
    # 显示结果  
    cv2.imshow('Rectangles', image)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    return num_row, num_col, boards_pos, boards_center


@torch.no_grad()
def parse_cv_boards(image, boards_pos, boards, st):
    num_row, num_col = boards_pos.shape[:2]
    for i in range(num_row):
        for j in range(num_col):
            pos = boards_pos[i][j]
            grid = image[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]]
            grid = cv2.resize(grid, (28, 28))
            cv2.imwrite(f'data/minesweeper/{st:03d}_{i:02d}_{j:02d}.jpg', grid)
    return boards


@torch.no_grad()
def parse_boards(image, boards_pos, boards, model, transform):
    num_row, num_col = boards.shape[:2]
    # new_img = np.zeros((num_row, num_col, 28, 28), dtype=np.uint8)
    for i in range(num_row):
        for j in range(num_col):
            if 0 <= boards[i][j] < 9 or boards[i][j] == 11:
                continue
            pos = boards_pos[i][j]
            grid = image[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]]
            grid = cv2.resize(grid, (28, 28))
            # new_img[i][j] = grid
            img = transform(grid)[None]
            out = model(img)[0].softmax(dim=-1)
            idx = torch.argmax(out).item()
            if out[idx] > 0.5:
                boards[i][j] = idx if idx < 7 else 10 + idx - 7
            if boards[i][j] == 12 or boards[i][j] == 14:
                print('failed')
                exit()
            
    # image = new_img.transpose(0, 2, 1, 3).reshape(num_row * 28, num_col * 28)
    # cv2.imwrite('debug/tmp.jpg', image)
    return boards


def click_sc(boards_center, r, c, button=pyautogui.PRIMARY, **kwargs):
    # pyautogui.MIDDLE
    # pyautogui.LEFT
    pyautogui.moveTo(boards_center[r, c, 0], boards_center[r, c, 1], duration=0.05)
    pyautogui.click(button=button)
    # pyautogui.click(boards_center[r, c, 0], boards_center[r, c, 1], button=button)



class MineSolver(object):
    def __init__(self, boards, ) -> None:
        self.boards = boards
        self.mine_boards = np.zeros_like(boards)
        self.num_row, self.num_col = boards.shape[:2]
        

    def check_grid(self, r, c):
        adj_cnt, unknow_cnt, mine_cnt = 0, 0, 0
        mine_pos, open_pos, unknow_pos = [], [], []
        for dy, dx in [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]:
            i = r + dy
            j = c + dx
            if 0 <= i < self.num_row and 0 <= j < self.num_col:
                adj_cnt += 1
                if self.boards[i][j] == 10 and self.mine_boards[i][j] == 0:
                    unknow_cnt += 1
                    unknow_pos.append((i, j))
                if self.boards[i][j] == 11 or self.mine_boards[i][j] == 1:
                    mine_cnt += 1
        # if r == 11 and c == 8:
        #     pdb.set_trace()
        if mine_cnt + unknow_cnt == self.boards[r][c]:
            for i, j in unknow_pos:
                mine_pos.append((i, j))
                self.mine_boards[i][j] = 1
                self.boards[i][j] == 11
        elif mine_cnt == self.boards[r][c]:
            for i, j in unknow_pos:
                open_pos.append((i, j))
        return mine_pos, open_pos, unknow_pos, unknow_cnt

    def flag_mine(self):
        boards = self.boards
        unknow_cnt = 0
        mine_pos, open_pos, unknow_pos = [], [], []
        for i in range(self.num_row):
            for j in range(self.num_col):
                if 0 < boards[i][j] < 9:
                    ret = self.check_grid(i, j)
                    mine_pos.extend(ret[0])
                    open_pos.extend(ret[1])
                    if ret[3] > 0:
                        unknow_pos.append((i, j))
                    unknow_cnt += ret[3]
        return set(mine_pos), set(open_pos), set(unknow_pos), unknow_cnt

    def check_grid2(self, r, c):
        mine_pos, unknow_pos = [], []
        for dy, dx in [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]:
            i = r + dy
            j = c + dx
            if 0 <= i < self.num_row and 0 <= j < self.num_col:
                if self.boards[i][j] == 10 and self.mine_boards[i][j] == 0:
                    unknow_pos.append((i, j))
                if self.boards[i][j] == 11 or self.mine_boards[i][j] == 1:
                    mine_pos.append((i, j))
        return set(mine_pos), set(unknow_pos)

    def check_grid_union(self, in_pose):
        in_pose = list(in_pose)
        mine_pos, open_pos, unknow_pos = [], [], []
        for i, j in in_pose:
            mine1, unknow1 = self.check_grid2(i, j)
            unknow_pos.extend(list(unknow1))
            for dy, dx in [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]:
                r = i + dy
                c = j + dx
                if (r, c) in in_pose:
                    # 计算交集
                    mine2, unknow2 = self.check_grid2(r, c)
                    unknow_union = unknow1 & unknow2
                    # 默认剩余的雷都在交集部分
                    unknow1_mine_cnt = self.boards[i][j] - len(mine1)
                    unknow2_mine_cnt = self.boards[r][c] - len(mine2)
                    unknow2_diff = unknow2 - unknow_union
                    if unknow2_mine_cnt - unknow1_mine_cnt == len(unknow2_diff):
                        mine_pos.extend(list(unknow2_diff))
                    # 计算交集内最少的雷数，若grid2中交集内雷数已经满足要求，则交集外都不是雷
                    unknow1_mine_cnt = self.boards[i][j] - len(mine1) - (len(unknow1) - len(unknow_union))
                    # 交集内最少有unknow1_mine_cnt个雷
                    if self.boards[r][c] - len(mine2) == unknow1_mine_cnt:
                        open_pos.extend(list(unknow2_diff))
        for i, j in mine_pos:
            self.mine_boards[i][j] = 1
            self.boards[i][j] == 11

        # 返回的unknow_poss随机选择
        return set(mine_pos), set(open_pos), set(unknow_pos) 

def main():
    win_pos = capture_window("扫雷")
    image = np.array(pyautogui.screenshot(region=win_pos))
    if image is None or np.prod(image.shape[:2]) < 10000:  
        print("Error: Image not found.")  
        exit()
    num_row, num_col, boards_pos, boards_center = get_board_pos(image)
    boards_center[..., 0] += win_pos[0]
    boards_center[..., 1] += win_pos[1]
    
    boards = np.zeros((num_row, num_col), dtype=np.int32) - 1
    # 识别模型
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    model = MinistNet()
    model.eval()
    model.load_state_dict(torch.load('ckpt/epoch29.pth'))
    # 点击中间
    time.sleep(1)
    click_sc(boards_center, num_row // 2, num_col // 2)
    time.sleep(0.05)
    solver = MineSolver(boards)
    
    for st in tqdm(range(100)):
        image = np.array(pyautogui.screenshot(region=win_pos))

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f'boards/step_{st:04d}.jpg', image)
        # parse_cv_boards(image, boards_pos, boards, st)
        boards = parse_boards(image, boards_pos, boards, model, transform)
        # if st == 10:
        #     pdb.set_trace()
        # 开棋盘, 插红旗， 1计算所有红旗位置
        mine_pos, open_pos, unknow_pos, unknow_cnt = solver.flag_mine()
        # 点击，计算所有可点击位置
        for pos in open_pos:
            click_sc(boards_center, pos[0], pos[1])
        for pos in mine_pos:
            click_sc(boards_center, pos[0], pos[1], button=pyautogui.RIGHT)
        if unknow_cnt == 0:
            break
        elif len(mine_pos) == 0 and len(open_pos) == 0:
            mine_pos, open_pos, unknow_pos = solver.check_grid_union(unknow_pos)
            for pos in open_pos:
                click_sc(boards_center, pos[0], pos[1])
            for pos in mine_pos:
                click_sc(boards_center, pos[0], pos[1], button=pyautogui.RIGHT)
            if len(mine_pos) == 0 and len(open_pos) == 0:
                # 随机选择一个
                print('random select')
                pos = list(unknow_pos)[0]
                click_sc(boards_center, pos[0], pos[1])
                # pdb.set_trace()
        # time.sleep(0.01)
        


if __name__ == '__main__':
    main()