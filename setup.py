from setuptools import setup, find_packages

install_reqs = ['torch==1.7.0', 'gym==0.17.0', 'pyglet==1.5.0', 'box2d-py']

setup(
        name='VR_REINFORCE',
        #url='https://github.com/anonymous/VR-REINFORCE',
        author='Anonymous',
        author_email='anonymous-email',
        #packages=find_packages(),
        #requierements
        install_requires=install_reqs,
        version='0.1',
        license='MIT',
        #keywords
        keywords=['Variance-reduced methods', 'ADP', 'Data-Driven Optimal Control', 'MDPs', 'model-free RL', 'policy gradient', 'REINFORCE-type methods'],
        #description of package
        description='An implementation of variance-reduced extensions of REINFORCE-type policy gradient methods for model-free RL.'
)
