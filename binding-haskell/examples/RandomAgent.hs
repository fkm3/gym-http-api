import Control.Monad (forM_)
import Control.Monad.Extra (whileM)
import Text.Printf (printf)
import qualified OpenAI.Gym as Gym

main :: IO ()
main = do
    Gym.withEnv' "http://127.0.0.1:5000" "CartPole-v0" $ \env ->
        forM_ [1..20] $ \i -> do
            printf "Starting episode %v.\n" (i :: Int)
            _ <- Gym.reset env
            whileM $ do
                action <- Gym.sample (Gym.actionSpace env)
                Gym.StepResult _ _ done <- Gym.step env action True
                return (not done)
    putStrLn "Done."
