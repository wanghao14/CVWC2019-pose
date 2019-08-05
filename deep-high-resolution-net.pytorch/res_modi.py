import json

ori_res_path = 'result/atrw/pose_hrnet/w48_384x288_OANW0731_test/results/keypoints_test_results_0.json'
formal_res_path = 'result/keypoints_test_results_0731_9.json'

formal_res = []
ori_res = json.load(open(ori_res_path, 'r'))

num_keypoints_statis = [0] * 15
for res in ori_res:
    num_keypoints = 0
    keypoints = []
    image_id = res['image_id']
    score = res['score']
    for i in range(15):
        if res['keypoints'][i * 3 + 2] > 0.20:
            if i != 14:
                keypoints.append(int('%d' % res['keypoints'][i * 3]))
                keypoints.append(int('%d' % res['keypoints'][i * 3 + 1]))
                keypoints.append(2)
            else:
                keypoints.append(int('%d' % ((res['keypoints'][2 * 3] +
                                             res['keypoints'][13 * 3]) / 2)))
                keypoints.append(int('%d' % ((res['keypoints'][2 * 3 + 1] +
                                             res['keypoints'][13 * 3 + 1]) / 2)))
                keypoints.append(2)
            num_keypoints += 1
        else:
            keypoints.extend([0, 0, 0])
    data_pack = {
        'image_id': image_id,
        'category_id': 1,
        'keypoints': keypoints,
        'num_keypoints': num_keypoints,
        'score': score
    }
    num_keypoints_statis[num_keypoints - 1] += 1
    formal_res.append(data_pack)
print(num_keypoints_statis)
with open(formal_res_path, 'w') as f:
    json.dump(formal_res, f, sort_keys=True, indent=4)
