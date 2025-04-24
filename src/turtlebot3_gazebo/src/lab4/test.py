from queue import Queue
import numpy as np


class WFD():
    def __init__(self):
        self.map_info = None
        # self.map_resolution = None
        self.map_width = 5
        self.map_height = 5
        self.map_origin = (2,3)
        self.occupancy_grid_2d = [
            [-1, -1, -1, -1, -1],
            [-1,  0,  0,  0, -1],
            [-1,  0, 100, 0, -1],
            [-1,  0,  0,  0, -1],
            [-1, -1, -1, -1, -1],
        ]  #[y][x]
        
    def wfd(self):
        # frontier selection (Wavefront frontier detection)
        # https://arxiv.org/abs/1806.03581
        # https://arxiv.org/pdf/1806.03581
        robot_x, robot_y = self.map_origin
        x_occup_idx = robot_x
        y_occup_idx = robot_y
        
        
        # UNLABELED = 0
        # MAP_OPEN_LIST = 1
        # MAP_CLOSE_LIST = 2
        # FRONTIER_OPEN_LIST = 3
        # FRONTIER_CLOSE_LIST = 4
        labeled_occupancy_grid_2d = np.zeros((5, 5), dtype=int)        #[y][x]
        
        # (x, y)
        outter_queue = Queue()
        frontier_list = list()
        
        # enqueue current robot position and mark as "MAP_OPEN_LIST"
        outter_queue.put((x_occup_idx, y_occup_idx))
        labeled_occupancy_grid_2d[y_occup_idx][x_occup_idx] = 1
        
        while not outter_queue.empty():
            x, y = outter_queue.get()
            
            if labeled_occupancy_grid_2d[y][x] == 2:
                continue
            
            if self.isFrontier(x, y):
                # (x, y)
                inner_queue = Queue()
                frontier_list_temp = list()
                
                inner_queue.put((x, y))
                labeled_occupancy_grid_2d[y][x] = 3
                
                while not inner_queue.empty():
                    x_i, y_i = inner_queue.get()
                    
                    if labeled_occupancy_grid_2d[y_i][x_i] == 2 or labeled_occupancy_grid_2d[y_i][x_i] == 4:
                        continue
                    
                    if self.isFrontier(x_i, y_i):
                        frontier_list_temp.append((x_i, y_i))
                        
                        neighbors = self.adjacentNeighbors(x_i, y_i)
                        if len(neighbors) > 0:
                            for x_ii, y_ii in neighbors:
                                
                                # [MAP_OPEN_LIST, FRONTIER_OPEN_LIST, FRONTIER_CLOSE_LIST]
                                if labeled_occupancy_grid_2d[y_ii][x_ii] not in [1, 3, 4]:
                                    inner_queue.put((x_ii, y_ii))
                                    labeled_occupancy_grid_2d[y_ii][x_ii] = 3

                    labeled_occupancy_grid_2d[y_i][x_i] = 4

                if len(frontier_list_temp) > 0:
                    frontier_list.append(frontier_list_temp)
                    for x_f, y_f in frontier_list_temp:
                        labeled_occupancy_grid_2d[y_f][x_f] = 2
            
            neighbors = self.adjacentNeighbors(x ,y)
            if len(neighbors) > 0:
                for x_n, y_n in neighbors:
                    if labeled_occupancy_grid_2d[y_n][x_n] not in [1,2] and self.freeNeighbor(x_n, y_n):
                        outter_queue.put((x_n, y_n))
                        labeled_occupancy_grid_2d[y_n][x_n] = 1
                        
            labeled_occupancy_grid_2d[y][x] = 2
            
        return frontier_list
                                           
        
    def isFrontier(self, x, y):
        # (x, y)
        if self.occupancy_grid_2d[y][x] != 0:
            return False
        
        neighbors = self.adjacentNeighbors(x, y)
                
        if len(neighbors) > 0:
            for x, y in neighbors:
                if self.occupancy_grid_2d[y][x] == -1:
                    return True 
        
        return False
    
    def adjacentNeighbors(self, x, y):
        # (x, y)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        neighbors = list()
        
        for dx, dy in directions:
            if (x + dx) >= 0 and (x + dx) < self.map_width and (y + dy) >= 0 and (y + dy) < self.map_height:
                neighbors.append((x + dx, y + dy))
        
        return neighbors
    
    def freeNeighbor(self, x, y):
        neighbors = self.adjacentNeighbors(x, y)
        if len(neighbors) > 0:
            for x_n, y_n in neighbors:
                if self.occupancy_grid_2d[y_n][x_n] == 0:
                    return True
    
        return False
    

def main():
    t = WFD()
    t.wfd()
    
if __name__ == "__main__":
    main()


    