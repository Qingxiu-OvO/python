import pygame
import random

# --- 配置 ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TILE_SIZE = 32
FPS = 60

# 颜色定义 (R, G, B)
WHITE = (255, 255, 255)
SKY_BLUE = (135, 206, 235)
DIRT_BROWN = (101, 67, 33)
GRASS_GREEN = (34, 139, 34)
STONE_GREY = (128, 128, 128)
PLAYER_COLOR = (255, 0, 0)  # 红色方块代表玩家
BLACK = (0, 0, 0)

# 方块ID定义
AIR = 0
DIRT = 1
GRASS = 2
STONE = 3

# --- 类定义 ---

class Player:
    def __init__(self, x, y, world):
        self.rect = pygame.Rect(x, y, 20, 30) # 玩家尺寸宽20高30
        self.vel_y = 0
        self.speed = 5
        self.jump_power = -12
        self.gravity = 0.5
        self.world = world
        self.grounded = False

    def move(self, left, right, jump):
        dx = 0
        dy = 0

        # 左右移动
        if left:
            dx = -self.speed
        if right:
            dx = self.speed

        # 跳跃
        if jump and self.grounded:
            self.vel_y = self.jump_power
            self.grounded = False

        # 重力
        self.vel_y += self.gravity
        if self.vel_y > 10:
            self.vel_y = 10
        
        dy += self.vel_y

        # --- 碰撞检测 (核心逻辑) ---
        
        # X轴碰撞
        self.rect.x += dx
        for tile in self.world.get_tiles_around(self.rect):
            if self.rect.colliderect(tile):
                if dx > 0: # 向右撞墙
                    self.rect.right = tile.left
                if dx < 0: # 向左撞墙
                    self.rect.left = tile.right

        # Y轴碰撞
        self.rect.y += dy
        self.grounded = False # 先假设在空中
        for tile in self.world.get_tiles_around(self.rect):
            if self.rect.colliderect(tile):
                if self.vel_y > 0: # 下落撞地
                    self.rect.bottom = tile.top
                    self.vel_y = 0
                    self.grounded = True
                if self.vel_y < 0: # 顶头
                    self.rect.top = tile.bottom
                    self.vel_y = 0

    def draw(self, screen):
        pygame.draw.rect(screen, PLAYER_COLOR, self.rect)

class World:
    def __init__(self):
        self.rows = SCREEN_HEIGHT // TILE_SIZE + 5
        self.cols = SCREEN_WIDTH // TILE_SIZE
        self.map_data = []
        self.generate()

    def generate(self):
        # 简单的地形生成
        for row in range(self.rows):
            layer = []
            for col in range(self.cols):
                if row < 5: # 顶部是天空
                    layer.append(AIR)
                elif row == 5: # 表面是草地
                    layer.append(GRASS)
                else: # 地下
                    # 随机生成石头或泥土
                    if random.random() < 0.15:
                        layer.append(STONE)
                    else:
                        layer.append(DIRT)
            self.map_data.append(layer)

    def draw(self, screen):
        for row in range(self.rows):
            for col in range(self.cols):
                tile_id = self.map_data[row][col]
                if tile_id != AIR:
                    color = WHITE
                    if tile_id == DIRT: color = DIRT_BROWN
                    elif tile_id == GRASS: color = GRASS_GREEN
                    elif tile_id == STONE: color = STONE_GREY
                    
                    pygame.draw.rect(screen, color, 
                                     (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                    # 画个边框好看点
                    pygame.draw.rect(screen, (0,0,0), 
                                     (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

    def get_tiles_around(self, rect):
        # 优化：只检测玩家周围的方块，而不是全图检测
        tiles = []
        col_start = max(0, rect.left // TILE_SIZE - 1)
        col_end = min(self.cols, rect.right // TILE_SIZE + 2)
        row_start = max(0, rect.top // TILE_SIZE - 1)
        row_end = min(self.rows, rect.bottom // TILE_SIZE + 2)

        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                if self.map_data[row][col] != AIR:
                    tile_rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                    tiles.append(tile_rect)
        return tiles

    def modify_tile(self, pos, action, selected_block):
        # action: 0 = dig (set to AIR), 1 = place
        col = pos[0] // TILE_SIZE
        row = pos[1] // TILE_SIZE

        if 0 <= row < self.rows and 0 <= col < self.cols:
            current_tile = self.map_data[row][col]
            
            # 挖掘
            if action == 0:
                self.map_data[row][col] = AIR
            
            # 放置 (只有当前位置是空气且不与玩家重叠时才能放置)
            elif action == 1:
                player_rect_logic = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                # 简单的防卡死：不能把方块放在玩家身体里
                if current_tile == AIR and not player.rect.colliderect(player_rect_logic):
                    self.map_data[row][col] = selected_block

# --- 主程序 ---

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Mini Terraria - Python Edition")
clock = pygame.time.Clock()

world = World()
player = Player(100, 100, world)

# 游戏状态
running = True
selected_block = DIRT # 默认手持泥土

while running:
    clock.tick(FPS)

    # --- 1. 事件处理 ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # 键盘数字键切换方块
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1: selected_block = DIRT
            if event.key == pygame.K_2: selected_block = GRASS
            if event.key == pygame.K_3: selected_block = STONE

    # 鼠标输入 (挖掘/建造)
    mouse_buttons = pygame.mouse.get_pressed()
    mouse_pos = pygame.mouse.get_pos()

    if mouse_buttons[0]: # 左键：挖掘
        world.modify_tile(mouse_pos, 0, None)
    if mouse_buttons[2]: # 右键：放置
        world.modify_tile(mouse_pos, 1, selected_block)

    # 键盘移动输入
    keys = pygame.key.get_pressed()
    player.move(keys[pygame.K_a] or keys[pygame.K_LEFT], 
                keys[pygame.K_d] or keys[pygame.K_RIGHT], 
                keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP])

    # --- 2. 绘制 ---
    screen.fill(SKY_BLUE) # 背景色
    
    world.draw(screen)
    player.draw(screen)

    # UI: 显示当前选中的方块
    ui_text = f"Current Block: {['Air', 'Dirt', 'Grass', 'Stone'][selected_block]} (Press 1-3)"
    font = pygame.font.SysFont('Arial', 20)
    text_surface = font.render(ui_text, True, BLACK)
    pygame.draw.rect(screen, WHITE, (5, 5, 300, 30)) # UI背景
    screen.blit(text_surface, (10, 10))

    pygame.display.update()

pygame.quit()