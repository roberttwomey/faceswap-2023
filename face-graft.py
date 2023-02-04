# adapted from example here: 
# https://google.github.io/mediapipe/solutions/face_mesh.html

import matplotlib.pyplot as plt
import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
import skimage
from skimage.transform import PiecewiseAffineTransform, warp

def imshow(image):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    return ax

uv_path = "./data/uv_map.json" #taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
uv_map_dict = json.load(open(uv_path))
uv_map = np.array([ (uv_map_dict["u"][str(i)],uv_map_dict["v"][str(i)]) for i in range(468)])
# borrowed from https://github.com/YadiraF/DECA/blob/f84855abf9f6956fb79f3588258621b363fa282c/decalib/utils/util.py

def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = np.array(verts)
    uvcoords = np.array(uvcoords)
    faces = np.array(faces); faces = faces.reshape(-1, 3) - 1
    uv_faces = np.array(uv_faces); uv_faces = uv_faces.reshape(-1, 3) - 1
    return (
        verts,
        uvcoords,
        faces,
        uv_faces
    )

# borrowed from https://github.com/YadiraF/DECA/blob/f84855abf9f6956fb79f3588258621b363fa282c/decalib/utils/util.py
def write_obj(obj_name,
              vertices,
              faces,
              texture_name = "texture.jpg",
              colors=None,
              texture=None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map=None,
              ):
    ''' Save 3D face model with texture. 
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
    '''
    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name
    material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write('f {}/{} {}/{} {}/{}\n'.format(
                    #  faces[i, 2], uvfaces[i, 2],
                    #  faces[i, 1], uvfaces[i, 1],
                    #  faces[i, 0], uvfaces[i, 0]
                    faces[i, 0], uvfaces[i, 0],
                    faces[i, 1], uvfaces[i, 1],
                    faces[i, 2], uvfaces[i, 2]
                )
                )
            # write mtl
            with open(mtl_name, 'w') as f:
                f.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
                f.write(s)

                if normal_map is not None:
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    f.write(f'disp {normal_name}')
                    # out_normal_map = normal_map / (np.linalg.norm(
                    #     normal_map, axis=-1, keepdims=True) + 1e-9)
                    # out_normal_map = (out_normal_map + 1) * 0.5

                    cv2.imwrite(
                        normal_name,
                        # (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
                        normal_map
                    )
            skimage.io.imsave(texture_name, texture)

# from https://www.appsloveworld.com/opencv/100/30/how-to-resize-image-and-maintain-aspect-ratio

def resizeToFit(image, height):
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def process_target(base_img, target, base_landmarks, targetlandmarks):
  img = source
  img2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_target = img.copy()
  img2_new_face = np.zeros_like(img)
  if len(targetlandmarks) > 0:
      points2 = np.array(targetlandmarks, np.int32)
      convexhull2 = cv2.convexHull(points2)

      triangle_handler = triangulate_faces(base_landmarks,  base_img)

      # for triangle_index in triangle_handler["index"]:
      #     # Triangulation of the first face
      #     tpt1 = base_face_handler["landmarks"][triangle_index[0]]
      #     tpt2 = base_face_handler["landmarks"][triangle_index[1]]
      #     tpt3 = base_face_handler["landmarks"][triangle_index[2]]
      #     triangle1 = np.array([tpt1, tpt2, tpt3], np.int32)

      #     rect1 = cv2.boundingRect(triangle1)
      #     (x, y, w, h) = rect1
      #     cropped_triangle = self.base_img[y: y + h, x: x + w]
      #     cropped_tr1_mask = np.zeros((h, w), np.uint8)

      #     points = np.array([[tpt1[0] - x, tpt1[1] - y],
      #                        [tpt2[0] - x, tpt2[1] - y],
      #                        [tpt3[0] - x, tpt3[1] - y]], np.int32)

      #     cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

      #     # Triangulation of second face
      #     t2pt1 = target_face_handler["landmarks"][triangle_index[0]]
      #     t2pt2 = target_face_handler["landmarks"][triangle_index[1]]
      #     t2pt3 = target_face_handler["landmarks"][triangle_index[2]]
      #     triangle2 = np.array([t2pt1, t2pt2, t2pt3], np.int32)

      #     rect2 = cv2.boundingRect(triangle2)
      #     (x, y, w, h) = rect2

      #     cropped_tr2_mask = np.zeros((h, w), np.uint8)

      #     points2 = np.array([[t2pt1[0] - x, t2pt1[1] - y],
      #                         [t2pt2[0] - x, t2pt2[1] - y],
      #                         [t2pt3[0] - x, t2pt3[1] - y]], np.int32)

      #     cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

      #     # Warp triangles
      #     points = np.float32(points)
      #     points2 = np.float32(points2)
      #     M = cv2.getAffineTransform(points, points2)
      #     warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h),borderMode=cv2.BORDER_REPLICATE)
      #     warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

      #     # Reconstructing destination face
      #     img2_new_face_rect_area = self.img2_new_face[y: y + h, x: x + w]
      #     img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
      #     _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
      #     warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

      #     img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
      #     self.img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
      # # Face swapped (putting 1st face into 2nd face)
      # img2_face_mask = np.zeros_like(img2_gray)
      # img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
      # # cv2.imshow("pabit", img2_head_mask)
      # img2_face_mask = cv2.bitwise_not(img2_head_mask)
      # seam_clone = img.copy()
      # self.img2_head_noface = cv2.bitwise_and(seam_clone, seam_clone, mask=img2_face_mask)

      # cv2.imshow("no_Head", self.img2_head_noface)
      # self.result = cv2.add(self.img2_head_noface, self.img2_new_face)

      # (x, y, w, h) = cv2.boundingRect(convexhull2)
      # center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

      # self.seamlessclone = cv2.seamlessClone(self.result, seam_clone,
      #                                   img2_head_mask, center_face2, cv2.MIXED_CLONE)

# ==== Read Input Image

#image source: https://mydramalist.com/people/461-aragaki-yui
# image_path = "./data/gakki.jpg"
# image_ori = skimage.io.imread(image_path)
# imshow(image_ori)

# load video
cap = cv2.VideoCapture('data/vollmann3.mov')
count = 0
trackedpoints = {}

# load webcap
cap2 = cv2.VideoCapture(0)

# save video
size = (512, 512)
result = cv2.VideoWriter('results/tracked.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)


tform = PiecewiseAffineTransform()
        
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# get keypoints
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
    
    face_mesh2 = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5)

    # loop over frames
    while cap.isOpened():
        ret,frame = cap.read()
        success, camimage = cap2.read()

        count = count + 1
        # cv2.imshow('window-name', frame)
        
        frame.flags.writeable = False
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H,W,_ = image.shape

        results = face_mesh.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

        # face_landmarks = results.multi_face_landmarks[0]
        # keypoints = np.array([(W*point.x,H*point.y) for point in face_landmarks.landmark[0:468]])#after 468 is iris or something else
        # ax = imshow(image)
        # ax.plot(keypoints[:, 0], keypoints[:, 1], '.b', markersize=2)
        # plt.show()

        # save to json file
        #https://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html
        # H_new,W_new = size#512,512
        # keypoints_uv = np.array([(W_new*x, H_new*y) for x,y in uv_map])

        # This step is slow
        # tform.estimate(keypoints_uv,keypoints)
        # texture = warp(image, tform, output_shape=(H_new,W_new))
        # texture = (255*texture).astype(np.uint8)

        # ax = imshow(texture)
        # ax.plot(keypoints_uv[:, 0], keypoints_uv[:, 1], '.b', markersize=2)
        # plt.show()
        # sys.stdout.write(".")
        # sys.stdout.flush()
        
        # result.write(texture)
            # Flip the image horizontally for a selfie-view display.

        # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        # Track live webcam feed
        # camimage.flags.writeable = False
        camimage = resizeToFit(camimage, 360)
        camimage = cv2.cvtColor(camimage, cv2.COLOR_BGR2RGB)
        H,W,_ = camimage.shape

        results2 = face_mesh2.process(camimage)
        camimage = cv2.cvtColor(camimage, cv2.COLOR_RGB2BGR)

        if results2.multi_face_landmarks:
          for face_landmarks in results2.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=camimage,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=camimage,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=camimage,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

        # do swapping
        draw_subimage(camimage, seamlessclone)

        comp1 = resizeToFit(image, 360)
        comp2 = resizeToFit(camimage, 360)

        composite = np.concatenate((comp2, comp1), axis=1)
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(composite, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break

cap.release()
cap2.release()
result.release()


# keypoints3d = np.array([(point.x,point.y,point.z) for point in face_landmarks.landmark[0:468]])
# obj_filename = "./data/canonical_face_model.obj"
# verts,uvcoords,faces,uv_faces = load_obj(obj_filename)

# def normalize_keypoints(keypoints3d):
#     center = keypoints3d[0]
#     keypoints3d = keypoints3d - center
#     axis1 = keypoints3d[165] - keypoints3d[391]
#     axis2 = keypoints3d[2] - keypoints3d[0]
#     axis3 = np.cross(axis2,axis1)
#     axis3 = axis3/np.linalg.norm(axis3)
#     axis2 = axis2/np.linalg.norm(axis2)
#     axis1 = np.cross(axis3, axis2)
#     axis1 = axis1/np.linalg.norm(axis1)
#     U = np.array([axis3,axis2,axis1])
#     keypoints3d = keypoints3d.dot(U)
#     keypoints3d = keypoints3d - keypoints3d.mean(axis=0)
#     return keypoints3d

# vertices = normalize_keypoints(keypoints3d)

# # borrowed from https://github.com/YadiraF/PRNet/blob/master/utils/write.py
# obj_name =  "./results/obj_model.obj"
# write_obj(obj_name,
#               vertices,
#               faces,
#               texture_name = "./results/texture.jpg",
#               texture=texture,
#               uvcoords=uvcoords,
#               uvfaces=uv_faces,
#               )