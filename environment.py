from __future__ import division


def create_env(env_id, args, rank=-1):
    if 'v0' in env_id:
        import ENV.DigitalPose2DBase as poseEnv
    else:  # 'v1' in env_id:
        import ENV.DigitalPose2D as poseEnv

    env = poseEnv.gym.make(env_id, args.render_save)

    return env
