import numpy as np
import pdb

def variable_fps_frame_selection(action_seq, high_fps_action, low_fps_action_counts, unique_action_ids, num_frames_to_sample=32, training=True):
    '''
    action_seq: sequence of action ids equal to the length of the video
    high_fps_action: selected action for which we must sample at a high FPS
    low_fps_action_counts: dictionary of counts defining how many frames each of the low FPS actions should have
    unique_action_ids: available unique action ids
    '''

    action_frames = []
    for act_id in unique_action_ids:
        frames = np.where(action_seq == act_id)[0]
        transition_idx = np.where(np.abs(np.diff(frames)) > 1)[0]
        if len(transition_idx) > 0:
            frames = frames[:transition_idx[0]+1]
        action_frames.append(frames)
    low_fps_frames = []
    num_low_fps_frames = 0
    for act_id in unique_action_ids:
        if act_id in low_fps_action_counts:
            if len(action_frames[act_id]) > 10:
                num_frames = low_fps_action_counts[act_id]
                start_frame = np.random.randint(0, 5) if training else 0
                frames = action_frames[act_id][np.round(np.linspace(start_frame, len(action_frames[act_id])-1, num_frames)).astype(int)]
                low_fps_frames.append(frames)
                num_low_fps_frames += len(frames)
            else:
                low_fps_frames.append(np.array([]))
        else:
            low_fps_frames.append([])
    if training and len(action_frames[high_fps_action]) > 10:
        start_frame = np.random.randint(0, 5)
    else:
        start_frame = 0
    high_fps_frames = action_frames[high_fps_action][np.round(np.linspace(start_frame, len(action_frames[high_fps_action])-1, num_frames_to_sample - num_low_fps_frames)).astype(int)]
    final_frames = []
    for act_id in unique_action_ids:
        if act_id == high_fps_action:
            final_frames.append(high_fps_frames)
        elif act_id in low_fps_action_counts:
            final_frames.append(low_fps_frames[act_id])
    selected_frames = np.concatenate(final_frames)
    return selected_frames

def constant_fps_frame_selection(action_seq, low_fps_action_counts, unique_action_ids, num_frames_to_sample=32, training=True):
    '''
    action_seq: sequence of action ids equal to the length of the video
    low_fps_action_counts: dictionary of counts defining how many frames each of the low FPS actions should have
    unique_action_ids: available unique action ids
    '''

    action_frames = []
    for act_id in unique_action_ids:
        if act_id in low_fps_action_counts:
            frames = np.where(action_seq == act_id)[0]
            transition_idx = np.where(np.abs(np.diff(frames)) > 1)[0]
            if len(transition_idx) > 0:
                frames = frames[:transition_idx[0]+1]
            action_frames.append(frames)
    action_frames = np.concatenate(action_frames)
    start_frame = np.random.randint(0, 5) if training else 0
    selected_frames = action_frames[np.round(np.linspace(start_frame, len(action_frames)-1, num_frames_to_sample)).astype(int)]
    return selected_frames


def non_action_aligned_variable_fps_frame_selection(total_frames, num_frames_to_sample=32, num_high_fps_frames=20, num_low_fps_frames=12):

    high_fps_start = np.random.randint(0, total_frames)
    # choose a random length between 32 frames and 30% of the full length of the video
    length_start = 32
    length_end = int(0.3 * total_frames)
    if length_end - length_start <= 10:
        length_start = 0
    random_length = np.random.randint(length_start, length_end)

    # choose start and end of high FPS frames
    high_fps_end = min(total_frames, high_fps_start + random_length)
    if high_fps_end - high_fps_start < random_length:
        high_fps_start = high_fps_end - random_length

    # choose 20 high FPS frames
    selected_high_fps_frames = np.round(np.linspace(high_fps_start, high_fps_end-1, num_high_fps_frames))

    remaining_frames = np.concatenate((np.array(list(range(0, high_fps_start))), np.array(list(range(high_fps_end, total_frames)))))

    # choose 12 low FPS frames
    selected_low_fps_frames = remaining_frames[np.round(np.linspace(0, len(remaining_frames)-1, num_low_fps_frames)).astype(int)]
    selected_frames = np.concatenate((selected_high_fps_frames, selected_low_fps_frames)).astype(int)
    selected_frames = np.sort(selected_frames)
    return selected_frames
