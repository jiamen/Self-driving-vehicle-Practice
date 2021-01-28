#ifndef PROJECTION_H
#define PROJECTION_H

#include "tools/rotation.h"

// camera : 9 dims array with 
// [0-2] : angle-axis rotation 
// [3-5] : translation
// [6-8] : camera parameter, [6] focal length焦距, [7-8] second and forth order radial distortion 畸变系数
// point : 3D location. 
// predictions : 2D predictions with center of the image plane. 

template <typename T>
inline bool CamProjectionWithDistortion(const T* camera, const T* point, T* predictions)
{
    // Rodrigues' formula
    T p[3];
    AngleAxisRotatePoint(camera, point, p);     // 将点由世界坐标系旋转到相机坐标系 R
    // camera[3,4,5] are the translation
    p[0] += camera[3];                          // 再进行平移，现在的点p是相机坐标系下的p
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center fo distortion         // 进行归一化
    T xp = -p[0]/p[2];
    T yp = -p[1]/p[2];

    // Apply second and fourth order radial distortion  取出畸变参数
    const T& l1 = camera[7];
    const T& l2 = camera[8];

    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2 * (l1 + l2 * r2);    // 相当于k1 k2两个参数 k1*r^2 + k2*r^4, 这个看第89页，得到系数

    const T& focal = camera[6];
    predictions[0] = focal * distortion * xp;       // x_distorted = xp * distortion
    predictions[1] = focal * distortion * yp;

    return true;
}



#endif // projection.h