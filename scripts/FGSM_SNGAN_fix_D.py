def train_gan_w_fgsm(dataloader, net_D, net_G, num_epochs, loss_fn, optim_G, optim_D, sched_G, sched_D, record, run_name=None):        
    if record:
        wandb.init(
        project="SNGAN_w_FGSM",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "betas": betas,
            "sheduler": "LambdaLR",
            "sheduler step": "1 - step / 100000",
            "sample step": 500,
            "FGSM epsilon": epsilon,
            "FGSM chance": fgsm_chance,
            "start FGSM epoch": start_fgsm_epoch
            })
        
        if run_name is not None:
            wandb.run.name = run_name
        
        fake = net_G(sample_z).cpu()
        grid = (make_grid(fake) + 1) / 2
        real, _ = next(iter(dataloader))
        grid2 = (make_grid(real[:64]) + 1) / 2
        images = wandb.Image(grid2, caption="real")
        wandb.log({"samples": images})
        images = wandb.Image(grid, caption="before training")
        wandb.log({"samples": images})
        
    # Training Loop    
    print("Starting Training Loop...")
    start = time.time()
    for epoch in range(1, num_epochs + 1):
        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader, desc=f'Training {epoch}/{num_epochs}')):
            # Discriminator
            with torch.no_grad():
                z = torch.randn(128, 100).to(device)
                fake = net_G(z).detach()
            real = data[0].to(device)
            real.requires_grad = True
            fake.requires_grad = True
            net_D_real = net_D(real)
            net_D_fake = net_D(fake)
            loss_D = loss_fn(net_D_real, net_D_fake)
            if epoch >= start_fgsm_epoch and np.random.random() < fgsm_chance:
                # FGSM
                real_grad = torch.autograd.grad(loss_D, [real])[0]
                fake_grad = torch.autograd.grad(loss_D, [fake])[0]
                perturbed_real = fgsm_attack(real, epsilon, real_grad)
                perturbed_fake = fgsm_attack(fake, epsilon, fake_grad)
                net_D_real = net_D(perturbed_real)
                net_D_fake = net_D(perturbed_fake)
                loss_D = loss_fn(net_D_real, net_D_fake)
            
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
            
            # Generator
            z = torch.randn(128 * 2, 100).to(device)
            if epoch >= start_fgsm_epoch and np.random.random() < fgsm_chance:
                # FGSM
                fake = net_G(z)
                net_D_fake = net_D(fake)
                loss_G = loss_fn(net_D_fake)
                fake_grad = torch.autograd.grad(loss_G, [fake])[0]
                perturbed_fake_G = fgsm_attack(fake, epsilon, fake_grad)
                loss_G = loss_fn(net_D(perturbed_fake_G))
            else:
                loss_G = loss_fn(net_D(net_G(z)))
                
            
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            sched_G.step()
            sched_D.step()
            
            if record:
                metrics = {"train/generator_loss": loss_G, 
                           "train/discriminator_loss": loss_D}
                wandb.log(metrics)
            
        print('loss_g = {:.3f}, loss_d = {:.3f}'.format(loss_G, loss_D))
        if record:
            fake = net_G(sample_z).detach()
            grid = (make_grid(fake.cpu()) + 1) / 2
            images = wandb.Image(grid, caption="after epoch {}".format(epoch))
            wandb.log({"samples": images})
            
            if epoch >= start_fgsm_epoch:
                # FGSM log
                grid = (make_grid(perturbed_real.cpu()) + 1) / 2
                images_real = wandb.Image(grid, caption="last in epoch {}".format(epoch))
                
                grid = (make_grid(perturbed_fake.cpu()) + 1) / 2
                images_fake = wandb.Image(grid, caption="last in epoch {}".format(epoch))
                
                grid = (make_grid(perturbed_fake_G.cpu()) + 1) / 2
                images_G = wandb.Image(grid, caption="last in epoch {}".format(epoch))
                
                wandb.log({
                    "FGSM on real in D": images_real,
                    "FGSM on fake in D": images_fake,
                    "FGSM in G": images_G
                })
            
            imgs = generate_imgs(
                    net_G, device, 100,
                    50000, batch_size)
            IS, FID = get_inception_score_and_fid(
                imgs, 'cifar10.train.npz', verbose=True)
            is_fid_imgs = {"train/IS": IS[0],
                          "train/IS_std": IS[1],
                          "train/FID:": FID}
            wandb.log(is_fid_imgs)

        if epoch % 5 == 0:
            clear_output()
        
    print('Training for batch size = {}, epochs = {} done for {:.1f} hours'.format(batch_size, num_epochs, (time.time() - start) / 60 / 60))
    wandb.finish()

