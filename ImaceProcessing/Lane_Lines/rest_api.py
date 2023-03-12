from flask import Flask, jsonify, request, render_template
import cv2
import base64
import numpy as np
from io import BytesIO

class LaneDetector:
    def __init__(self):
        pass
    
    def detect_lane_lines(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to smooth the image and reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection to detect edges in the image
        edges = cv2.Canny(blur, 50, 150)

        # Create a region of interest (ROI) mask to focus on the lane lines
        height, width = image.shape[:2]
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (width / 2, height / 2),
            (width, height),
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Apply Hough line detection to find the lane lines
        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
        
        """
        output_image = cv2.addWeighted(image, 0.8, lines, 1, 0)
        cv2.imshow('Lane Lines', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        
        # Separate the left lane line and right lane line based on their slopes
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)
                if slope < 0: # Left lane line
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                elif slope > 0: # Right lane line
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        # Fit a polynomial curve to the left lane line and right lane line
        if len(left_line_x) > 0 and len(left_line_y) > 0 and len(right_line_x) > 0 and len(right_line_y) > 0:
            left_fit = np.polyfit(left_line_y, left_line_x, 1)
            right_fit = np.polyfit(right_line_y, right_line_x, 1)
            left_line = self._get_line_points(left_fit, height)
            right_line = self._get_line_points(right_fit, height)
            return left_line, right_line
        else:
            return None, None
    
    def draw_lane_lines(self, image):
        # Detect lane lines
        left_line, right_line = self.detect_lane_lines(image)

        # Draw lane lines on original image
        if left_line is not None and right_line is not None:
            lines_image = np.zeros_like(image)
            cv2.line(lines_image, left_line[0], left_line[1], (0, 255, 0), 10)
            cv2.line(lines_image, right_line[0], right_line[1], (0, 255, 0), 10)
            return cv2.addWeighted(image, 0.8, lines_image, 1, 0)
        else:
            return image
    
    def draw_dotted_lines(self, image):
        # Detect lane lines
        left_line, right_line = self.detect_lane_lines(image)

        # Draw dotted lines on original image
        if left_line is not None and right_line is not None:
            lines_image = np.zeros_like(image)
            line_color = (0, 255, 0)
            line_thickness = 10
            line_gap = 20
            
            # Draw left lane line
            for y in range(image.shape[0], 0, -line_gap):
                x = int((y - left_line[1]) / left_line[0])
                cv2.line(lines_image, (x, y), (x, y - line_gap // 2), line_color, line_thickness)
            
            # Draw right lane line
            for y in range(image.shape[0], 0, -line_gap):
                x = int((y - right_line[1]) / right_line[0])
                cv2.line(lines_image, (x, y), (x, y - line_gap // 2), line_color, line_thickness)
            
            return cv2.addWeighted(image, 0.8, lines_image, 1, 0)
        else:
            return image
    
    def _get_line_points(self, line_fit, height):
        slope, intercept = line_fit
        y1 = height
        y2 = int(height / 2)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1), (x2, y2)

class ImageProcessor:
    def __init__(self):
        self.detector = LaneDetector()

    def process_image(self, image_data):
        # Convert image bytes to numpy array
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        # Decode image
        img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
        # Detect lane lines and draw dotted lines
        result = self.detector.draw_dotted_lines(img)
        # Encode image as JPEG
        ret, jpeg = cv2.imencode('.jpg', result)
        # Convert JPEG bytes to Base64-encoded string
        b64_string = base64.b64encode(jpeg).decode()
        return b64_string

app = Flask(__name__)
processor = ImageProcessor()

@app.route('/api/get-image', methods=['GET'])
def get_image():
    # Load image from file
    image = cv2.imread('image.jpg')
    # Detect lane lines and draw dotted lines
    result = processor.detector.draw_dotted_lines(image)
    # Encode image as JPEG
    ret, jpeg = cv2.imencode('.jpg', result)
    # Convert JPEG bytes to bytes buffer
    buffer = BytesIO(jpeg.tobytes())
    # Return image buffer
    return buffer.getvalue(), 200, {'Content-Type': 'image/jpeg'}

@app.route('/api/process-image', methods=['POST'])
def process_image():
    # Read image bytes from request
    image_data = request.get_data()
    # Process image
    result = processor.process_image(image_data)
    # Return processed image data as Base64-encoded string
    return jsonify({'image_data': result})

@app.route('/')
def index():
    return render_template('index3.html', image_data="image.jpeg")

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run()