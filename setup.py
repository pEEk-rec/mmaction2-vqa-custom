# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import sys
import warnings
from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'mmaction/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file.
    
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    
    Returns:
        List[str]: list of requirements items
    """
    import re
    from os.path import exists
    
    require_fpath = fname
    
    # Return empty list if file doesn't exist
    if not exists(require_fpath):
        warnings.warn(f'Requirements file {require_fpath} not found. Skipping.')
        return []
    
    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]
                
                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info
    
    def parse_require_file(fpath):
        if not exists(fpath):
            return
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info
    
    def gen_packages_items():
        for info in parse_require_file(require_fpath):
            parts = [info['package']]
            if with_version and 'version' in info:
                parts.extend(info['version'])
            if not sys.version.startswith('3.4'):
                platform_deps = info.get('platform_deps')
                if platform_deps is not None:
                    parts.append(';' + platform_deps)
            item = ''.join(parts)
            yield item
    
    packages = list(gen_packages_items())
    return packages


def add_mim_extension():
    """Add extra files that are required to support MIM into the package."""
    
    # parse installment mode
    if 'develop' in sys.argv:
        mode = 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        mode = 'copy'
    else:
        return
    
    filenames = ['tools', 'configs', 'model-index.yml', 'dataset-index.yml']
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'mmaction', '.mim')
    os.makedirs(mim_path, exist_ok=True)
    
    for filename in filenames:
        if osp.exists(filename):
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(mim_path, filename)
            
            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)
            
            if mode == 'symlink':
                src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                try:
                    os.symlink(src_relpath, tar_path)
                except OSError:
                    mode = 'copy'
                    warnings.warn(
                        f'Failed to create symbolic link for {src_relpath}, '
                        f'copying to {tar_path}')
                else:
                    continue
            
            if mode == 'copy':
                if osp.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)
                else:
                    warnings.warn(f'Cannot copy file {src_path}.')


if __name__ == '__main__':
    add_mim_extension()
    
    # Build extras_require dict, skipping missing files
    extras_require = {}
    extra_files = {
        'all': 'requirements.txt',
        'tests': 'requirements/tests.txt',
        'optional': 'requirements/optional.txt',
        'mim': 'requirements/mminstall.txt',
        'multimodal': 'requirements/multimodal.txt',
    }
    
    for key, fname in extra_files.items():
        reqs = parse_requirements(fname)
        if reqs:  # Only add if requirements exist
            extras_require[key] = reqs
    
    setup(
        name='mmaction2',
        version=get_version(),
        description='OpenMMLab Video Understanding Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='MMAction2 Contributors',
        author_email='openmmlab@gmail.com',
        maintainer='MMAction2 Contributors',
        maintainer_email='openmmlab@gmail.com',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        keywords='computer vision, video understanding',
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        url='https://github.com/open-mmlab/mmaction2',
        license='Apache License 2.0',
        install_requires=parse_requirements('requirements/build.txt'),
        extras_require=extras_require,  # Fixed: built dynamically
        zip_safe=False
    )
