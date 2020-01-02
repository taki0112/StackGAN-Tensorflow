from ops import *
from utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np


class StackGAN():
    def __init__(self, sess, args):

        self.phase = args.phase
        self.model_name = 'StackGAN'

        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_iter = args.decay_iter

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr

        self.gan_type = args.gan_type

        self.condition_dim = 128
        self.df_dim = 96
        self.gf_dim = 128
        self.text_dim = 1024
        self.z_dim = 100


        """ Weight """
        self.adv_weight = args.adv_weight
        self.kl_weight = args.kl_weight


        """ Generator """

        """ Discriminator """
        self.sn = args.sn

        self.img_height = args.img_height
        self.img_width = args.img_width

        self.img_ch = args.img_ch

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.dataset_path = os.path.join('./dataset', self.dataset_name)

        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)

        print()

        print("##### Generator #####")

        print()

        print("##### Discriminator #####")
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# kl_weight : ", self.kl_weight)

        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def generator_1(self, text_embedding, noise, is_training=True, reuse=tf.AUTO_REUSE, scope='generator_1'):
        channels = self.gf_dim * 8 # 1024
        with tf.variable_scope(scope, reuse=reuse):
            mu = fully_connected(text_embedding, units=self.condition_dim, use_bias=True, sn=self.sn, scope='mu_fc')
            mu = relu(mu)

            logvar = fully_connected(text_embedding, units=self.condition_dim, use_bias=True, sn=self.sn, scope='logvar_fc')
            logvar = relu(logvar)

            condition = reparametrize(mu, logvar)

            z = tf.concat([noise, condition], axis=-1)
            z = fully_connected(z, units=channels * 4 * 4, use_bias=False, sn=self.sn)
            z = batch_norm(z, is_training)
            z = relu(z)
            z = tf.reshape(z, shape=[-1, 4, 4, channels])

            x = z
            for i in range(4) :
                x = up_block(x, channels=channels // 2, is_training=is_training, use_bias=False, sn=self.sn, scope='up_block_' + str(i))
                channels = channels // 2

            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, scope='g_logit')
            x = tanh(x)

            return x, mu, logvar

    def generator_2(self, x_init, text_embedding, is_training=True, reuse=tf.AUTO_REUSE, scope='generator_2'):
        channels = self.gf_dim
        with tf.variable_scope(scope, reuse=reuse):

            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, scope='conv')
            x = relu(x)

            for i in range(2):
                x = conv(x, channels * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                x = relu(x)

                channels = channels * 2

            mu = fully_connected(text_embedding, units=self.condition_dim, use_bias=True, sn=self.sn, scope='mu_fc')
            mu = relu(mu)

            logvar = fully_connected(text_embedding, units=self.condition_dim, use_bias=True, sn=self.sn, scope='logvar_fc')
            logvar = relu(logvar)

            condition = reparametrize(mu, logvar)
            condition = tf.reshape(condition, shape=[-1, 1, 1, self.condition_dim])
            condition = tf.tile(condition, multiples=[1, 16, 16, 1])

            x = tf.concat([x, condition], axis=-1)

            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='joint_conv')
            x = batch_norm(x, is_training, scope='joint_batch_norm')
            x = relu(x)

            for i in range(2):
                x = resblock(x, channels, is_training, use_bias=False, sn=self.sn, scope='resblock_' + str(i))

            for i in range(4):
                x = up_block(x, channels=channels // 2, is_training=is_training, use_bias=False, sn=self.sn, scope='up_block_' + str(i))
                channels = channels // 2

            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, scope='g_logit')
            x = tanh(x)

            return x, mu, logvar

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator_1(self, x_init, mu, is_training=True, reuse=tf.AUTO_REUSE, scope="discriminator_1"):
        channel = self.df_dim

        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, scope='conv')
            x = lrelu(x, 0.2)

            for i in range(3) :
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            mu = tf.reshape(mu, shape=[-1, 1, 1, self.condition_dim])
            mu = tf.tile(mu, multiples=[1, 4, 4, 1])

            x = tf.concat([x, mu], axis=-1)

            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_last')
            x = batch_norm(x, is_training, scope='batch_norm_last')
            x = lrelu(x, 0.2)

            x = conv(x, channels=1, kernel=4, stride=4, use_bias=True, sn=self.sn, scope='d_logit')

            return x

    def discriminator_2(self, x_init, mu, is_training=True, reuse=tf.AUTO_REUSE, scope="discriminator_2"):
        channel = self.df_dim

        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, scope='conv')
            x = lrelu(x, 0.2)

            for i in range(5) :
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            for i in range(2):
                x = conv(x, channel // 2, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv3x3_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm3x3_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel // 2

            mu = tf.reshape(mu, shape=[-1, 1, 1, self.condition_dim])
            mu = tf.tile(mu, multiples=[1, 4, 4, 1])

            x = tf.concat([x, mu], axis=-1)

            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_last')
            x = batch_norm(x, is_training, scope='batch_norm_last')
            x = lrelu(x, 0.2)

            x = conv(x, channels=1, kernel=4, stride=4, use_bias=True, sn=self.sn, scope='d_logit')

            return x

    ##################################################################################
    # Model
    ##################################################################################


    def build_model(self):

        if self.phase == 'train' :
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            """ Input Image"""
            img_data_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.augment_flag)
            img_data_class.preprocess()

            self.dataset_num = len(img_data_class.image_list)


            img_and_embedding = tf.data.Dataset.from_tensor_slices((img_data_class.image_list, img_data_class.embedding))

            gpu_device = '/gpu:0'
            img_and_embedding = img_and_embedding.apply(shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing, batch_size=self.batch_size, num_parallel_batches=16,
                              drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))


            img_and_embedding_iterator = img_and_embedding.make_one_shot_iterator()

            self.real_img_256, self.embedding = img_and_embedding_iterator.get_next()
            sentence_index = tf.random.uniform(shape=[], minval=0, maxval=10, dtype=tf.int32)
            self.embedding = tf.gather(self.embedding, indices=sentence_index, axis=1) #[bs, 1024]

            noise = tf.random_normal(shape=[self.batch_size, self.z_dim])
            self.fake_img_64, mu_64, logvar_64 = self.generator_1(self.embedding, noise)
            self.fake_img_256, mu_256, logvar_256 = self.generator_2(self.fake_img_64, self.embedding)
            self.real_img_64 = tf.image.resize_bilinear(self.real_img_256, size=[64, 64])

            self.real_img = [self.real_img_64, self.real_img_256]
            self.fake_img = [self.fake_img_64, self.fake_img_256]

            real_logit_64 = self.discriminator_1(self.real_img_64, mu_64)
            fake_logit_64 = self.discriminator_1(self.fake_img_64, mu_64)

            real_logit_256 = self.discriminator_2(self.real_img_256, mu_256)
            fake_logit_256 = self.discriminator_2(self.fake_img_256, mu_256)

            g_adv_loss_64 = generator_loss(self.gan_type, fake_logit_64) * self.adv_weight
            g_kl_loss_64 = kl_loss(mu_64, logvar_64) * self.kl_weight

            d_adv_loss_64 = discriminator_loss(self.gan_type, real_logit_64, fake_logit_64) * self.adv_weight

            g_loss_64 = g_adv_loss_64 + g_kl_loss_64
            d_loss_64 = d_adv_loss_64

            g_adv_loss_256 = generator_loss(self.gan_type, fake_logit_256) * self.adv_weight
            g_kl_loss_256 = kl_loss(mu_256, logvar_256) * self.kl_weight

            d_adv_loss_256 = discriminator_loss(self.gan_type, real_logit_256, fake_logit_256) * self.adv_weight

            g_loss_256 = g_adv_loss_256 + g_kl_loss_256
            d_loss_256 = d_adv_loss_256

            self.g_loss = [g_loss_64, g_loss_256]
            self.d_loss = [d_loss_64, d_loss_256]


            """ Training """
            t_vars = tf.trainable_variables()
            G1_vars = [var for var in t_vars if 'generator_1' in var.name]
            G2_vars = [var for var in t_vars if 'generator_2' in var.name]
            D1_vars = [var for var in t_vars if 'discriminator_1' in var.name]
            D2_vars = [var for var in t_vars if 'discriminator_2' in var.name]

            g1_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(g_loss_64, var_list=G1_vars)
            g2_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(g_loss_256, var_list=G2_vars)

            d1_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(d_loss_64,var_list=D1_vars)
            d2_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(d_loss_256, var_list=D2_vars)

            self.g_optim = [g1_optim, g2_optim]
            self.d_optim = [d1_optim, d2_optim]


            """" Summary """
            self.summary_g_loss_64 = tf.summary.scalar("g_loss_64", g_loss_64)
            self.summary_g_loss_256 = tf.summary.scalar("g_loss_256", g_loss_256)
            self.summary_d_loss_64 = tf.summary.scalar("d_loss_64", d_loss_64)
            self.summary_d_loss_256 = tf.summary.scalar("d_loss_256", d_loss_256)

            self.summary_g_adv_loss_64 = tf.summary.scalar("g_adv_loss_64", g_adv_loss_64)
            self.summary_g_adv_loss_256 = tf.summary.scalar("g_adv_loss_256", g_adv_loss_256)
            self.summary_g_kl_loss_64 = tf.summary.scalar("g_kl_loss_64", g_kl_loss_64)
            self.summary_g_kl_loss_256 = tf.summary.scalar("g_kl_loss_256", g_kl_loss_256)

            self.summary_d_adv_loss_64 = tf.summary.scalar("d_adv_loss_64", d_adv_loss_64)
            self.summary_d_adv_loss_256 = tf.summary.scalar("d_adv_loss_256", d_adv_loss_256)


            g_summary_list = [self.summary_g_loss_64, self.summary_g_loss_256,
                              self.summary_g_adv_loss_64, self.summary_g_adv_loss_256,
                              self.summary_g_kl_loss_64, self.summary_g_kl_loss_256]

            d_summary_list = [self.summary_d_loss_64, self.summary_d_loss_256,
                              self.summary_d_adv_loss_64, self.summary_d_adv_loss_256]

            self.summary_merge_g_loss = tf.summary.merge(g_summary_list)
            self.summary_merge_d_loss = tf.summary.merge(d_summary_list)

        else :
            """ Test """
            """ Input Image"""
            img_data_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, augment_flag=False)
            img_data_class.preprocess()

            self.dataset_num = len(img_data_class.image_list)

            img_and_embedding = tf.data.Dataset.from_tensor_slices(
                (img_data_class.image_list, img_data_class.embedding))

            gpu_device = '/gpu:0'
            img_and_embedding = img_and_embedding.apply(shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing, batch_size=5, num_parallel_batches=16,
                              drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

            img_and_embedding_iterator = img_and_embedding.make_one_shot_iterator()

            self.real_img_256, self.embedding = img_and_embedding_iterator.get_next()
            sentence_index = tf.random.uniform(shape=[], minval=0, maxval=10, dtype=tf.int32)
            self.embedding = tf.gather(self.embedding, indices=sentence_index, axis=1)  # [bs, 1024]

            noise = tf.random_normal(shape=[self.batch_size, self.z_dim])
            self.fake_img_64, mu_64, logvar_64 = self.generator_1(self.embedding, noise, is_training=False)
            self.fake_img_256, mu_256, logvar_256 = self.generator_2(self.fake_img_64, self.embedding, is_training=False)

            self.test_fake_img = self.fake_img_256
            self.test_real_img = self.real_img_256


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=10)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_batch_id = checkpoint_counter
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")

        else:
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()

        for stage in range(2) :
            lr = self.init_lr
            for idx in range(start_batch_id, self.iteration):

                if self.decay_flag :
                    if idx > 0 and (idx % self.decay_iter) == 0 :
                        lr = self.init_lr * pow(0.5, idx // self.decay_iter)

                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.d_optim[stage], self.d_loss[stage], self.summary_merge_d_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                real_images, fake_images, _, g_loss, summary_str = self.sess.run(
                    [self.real_img[stage], self.fake_img[stage],
                     self.g_optim[stage],
                     self.g_loss[stage], self.summary_merge_g_loss], feed_dict=train_feed_dict)

                self.writer.add_summary(summary_str, counter)


                # display training status
                counter += 1
                print("Stage: [%1d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (stage, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx + 1, self.print_freq) == 0:
                    real_images = real_images[:5]
                    fake_images = fake_images[:5]

                    merge_real_images = np.expand_dims(return_images(real_images, [5, 1]), axis=0)
                    merge_fake_images = np.expand_dims(return_images(fake_images, [5, 1]), axis=0)

                    merge_images = np.concatenate([merge_real_images, merge_fake_images], axis=0)

                    save_images(merge_images, [1, 2],
                                './{}/merge_stage{}_{:07d}.jpg'.format(self.sample_dir, stage, idx + 1))


                if np.mod(counter - 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}_{}adv_{}kl{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                           self.adv_weight, self.kl_weight,
                                                           sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparisondkssjg
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>content</th><th>style</th><th>output</th></tr>")

        real_images, fake_images = self.sess.run([self.test_real_img, self.test_fake_img])
        for i in range(5) :
            real_path = os.path.join(self.result_dir, 'real_{}.jpg'.format(i))
            fake_path = os.path.join(self.result_dir, 'fake_{}.jpg'.format(i))

            real_image = np.expand_dims(real_images[i], axis=0)
            fake_image = np.expand_dims(fake_images[i], axis=0)

            save_images(real_image, [1, 1], real_path)
            save_images(fake_image, [1, 1], fake_path)

            index.write("<td>%s</td>" % os.path.basename(real_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (real_path if os.path.isabs(real_path) else (
                    '../..' + os.path.sep + real_path), self.img_width, self.img_height))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (fake_path if os.path.isabs(fake_path) else (
                    '../..' + os.path.sep + fake_path), self.img_width, self.img_height))
            index.write("</tr>")

        index.close()