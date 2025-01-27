import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from matplotlib.animation import FuncAnimation

# Load the trained YOLO model
model = YOLO('best.pt')  # Ensure 'best.pt' is in the correct path

def display_image_with_detections(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    results = model(image_path)
    result_image = results[0].plot()
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    return result_image_rgb, results

def generate_path_to_target(start, target, num_steps=100):
    path_x, path_y = [start[0]], [start[1]]
    current_point = np.array(start, dtype=np.float64)
    step_vector = (target - start) / num_steps

    for _ in range(num_steps):
        current_point += step_vector
        path_x.append(current_point[0])
        path_y.append(current_point[1])

    return path_x, path_y

def extract_obstacles(results):
    obstacles = []
    for box in results[0].boxes:
        x_center = (box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2
        y_center = (box.xyxy[0][1].item() + box.xyxy[0][3].item()) / 2
        obstacles.append((x_center, y_center))
    return obstacles

def animate_paths(i, all_paths, lines, annotations, target_reached, shortest_target_index):
    for idx, (path, line) in enumerate(zip(all_paths, lines)):
        max_index = len(path[0]) - 1

        # Update line data to current frame
        if i <= max_index:
            line.set_data(path[0][:i+1], path[1][:i+1])

        # When target is reached, set target_reached and make the annotation visible
        if i == max_index:
            target_reached[idx] = True
            annotations[idx].set_visible(True)  # Show annotation when target is reached
            
            # If this is not the shortest path, clear the line after reaching target
            if idx != shortest_target_index:
                line.set_data([], [])  # Hide the orange path line after reaching its target

    return lines + annotations

def main():
    image_path = '123.jpg'
    result_image_rgb, results = display_image_with_detections(image_path)
    if results is None:
        return
    obstacles = extract_obstacles(results)
    start = np.array([300, 600])
    targets = [np.array([100, 100]), np.array([400, 100]), np.array([200, 100]), np.array([300, 100])]

    distances = [np.linalg.norm(start - target) for target in targets]
    shortest_target_index = np.argmin(distances)
    speed = 5.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(result_image_rgb)
    if obstacles:
        ax.scatter(*zip(*obstacles), color='red', label='Marked Objects', s=100)
    ax.scatter(start[0], start[1], color='green', label='Start', s=100)

    all_paths = []
    lines = []
    target_reached = [False] * len(targets)
    annotations = []

    for i, target in enumerate(targets):
        path_x, path_y = generate_path_to_target(start, target)
        all_paths.append((path_x, path_y))
        distance = distances[i]
        time_taken = distance / speed

        # Initial invisible annotation, to be shown when the target is reached
        annotation = ax.annotate(f'Distance: {distance:.2f}\nTime: {time_taken:.2f}s', 
                                 (target[0], target[1]), textcoords="offset points", 
                                 xytext=(10, 20), ha='center', color='black', fontsize=5,
                                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        annotation.set_visible(False)  # Hide initially
        annotations.append(annotation)

        color = 'green' if i == shortest_target_index else 'orange'
        line, = ax.plot([], [], label=f'Target ({target[0]}, {target[1]})', linewidth=2, color=color)
        lines.append(line)

    ax.set_title('Detection Results and Paths to Multiple Targets')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)

    ani = FuncAnimation(fig, animate_paths, frames=len(all_paths[0][0]), 
                        fargs=(all_paths, lines, annotations, target_reached, shortest_target_index), 
                        interval=100, repeat=False, blit=True)
    
    plt.show()

if __name__ == '__main__':
    main()
