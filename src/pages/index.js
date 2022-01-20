/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';

import Layout from '@theme/Layout';

import classnames from 'classnames';

import styles from './styles.module.css';

/** Won't render children on server */
function ClientOnly({children, fallback}) {
  if (typeof window === 'undefined') {
    return fallback || null;
  }
  return children;
}

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;

  return (
    <Layout permalink="/" description={siteConfig.tagline}>
      <div className="hero hero--primary shadow--lw">
        <div className="container">
          <div className="row">
            <div className="col">
              <h1 className="hero__title">{siteConfig.title}</h1>
              <p className="hero__subtitle">{siteConfig.tagline}</p>
              <div>
                <Link
                  className="button button--secondary button--lg"
                  to={'https://github.com/facebookresearch/AugLy'}>
                  Get Started
                </Link>
              </div>
            </div>
            <div className="col text--center">
              <img
                className={styles.demoPic}
                src={useBaseUrl('/img/image_visual.png')}
              />
            </div>
          </div>
        </div>
      </div>
      <div className="container">
        <div className="margin-vert--xl">
          <div className="row">
            <div className="col">
              <h3>Real-World Augmentations</h3>
              <p>
                We offer more than 100 data augmentations focused on things that real people on the Internet do to images and videos on platforms like Facebook and Instagram.
              </p>
            </div>
            <div className="col">
              <h3>Extensive and Customizable</h3>
              <p>
                Augmentations can include a wide variety of modifications to a piece of content, ranging from recropping a photo to changing the pitch of a voice recording.
              </p>
            </div>
            <div className="col">
              <h3>Multimodal Data</h3>
              <p>
                Combining different modalities -- such as text and images or audio and video -- using real-world augmentations can help machines 
better understand complex content.
              </p>
            </div>
          </div>
        </div>
        <div className="margin-vert--xl text--center">
          <Link
            className="button button--primary button--lg"
            to={'https://ai.facebook.com/blog/augly-a-new-data-augmentation-library-to-help-build-more-robust-ai-models/'}>
            Learn more about AugLy
          </Link>
        </div>
      </div>
    </Layout>
  );
}

export default Home;
