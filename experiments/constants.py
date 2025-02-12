HU_FACTOR = 32768

WINDOWS = ['lung', 'abdomen', 'bone']
ALL_WIN = {'center': 500, 'width': 3000}
LUNG_WIN = {'center': -500, 'width': 1400}
ABD_WIN = {'center': 50, 'width': 400}
BONE_WIN = {'center': 400, 'width': 1500}

LESION_TYPES = {
    -1: None,
    1: 'bone',
    2: 'abdomen',
    3: 'mediastinum',
    4: 'liver',
    5: 'lung',
    6: 'kidney',
    7: 'soft tissue',
    8: 'pelvis'
}