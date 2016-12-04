{-# LANGUAGE OverloadedStrings #-}
module OpenAI.Gym
    ( Client
    , withClient
    , Env
    , withEnv
    , withEnv'
    , Space(..)
    , actionSpace
    , observationSpace
    , sample
    , reset
    , StepResult(..)
    , step
    ) where

import Control.Exception (bracket)
import Control.Lens (Prism', iso, (^..), (^.), (^?))
import Control.Monad (zipWithM)
import Data.Aeson.Lens (key, values, _String, _Bool, _Integral, _Value, _Number, AsNumber, _Double)
import Data.Int (Int64)
import Data.Maybe (fromMaybe)
import Data.Monoid ((<>))
import System.Random (randomRIO)
import Text.Printf (printf)
import qualified Data.Aeson as Aeson
import qualified Data.ByteString as B
import qualified Data.HashMap.Strict as HMap
import qualified Data.Scientific as Scientific
import qualified Data.Text as Text
import qualified Data.Vector.Storable as S
import qualified Network.Wreq as Wreq
import qualified Network.Wreq.Session as WreqS

-- TODO: Use multi-dimensional arrays instead of S.Vector.

data Client = Client
    { _clientURL :: String
    , _clientSession :: WreqS.Session
    } deriving (Show)

-- | Connect to a gym-http-api server.
withClient :: String  -- ^ URL.
           -> (Client -> IO a)
           -> IO a
withClient url f = WreqS.withSession (f . Client url)

data Env = Env
    { _envClient :: Client
    , _envInstanceID :: String
    , actionSpace :: Space
    , observationSpace :: Space
    } deriving (Show)

-- | Create an instance of a gym environment by ID, e.g. "CartPole-v0".
withEnv :: Client -> String -> (Env -> IO a) -> IO a
withEnv client envID = bracket (createEnv client envID) destroyEnv

createEnv :: Client -> String -> IO Env
createEnv client@(Client url sess) envID = do
    let reqURL = url <> "/v1/envs/"
        reqBody = Aeson.Object $
            HMap.singleton "env_id" (Aeson.String (Text.pack envID))
    r <- Wreq.asJSON =<< WreqS.post sess reqURL reqBody
    let body = r ^. Wreq.responseBody :: Aeson.Value
    let instanceID = Text.unpack (body ^. key "instance_id" . _String)
    aSpace <- getSpace client instanceID "/action_space/"
    oSpace <- getSpace client instanceID "/observation_space/"
    return (Env client instanceID aSpace oSpace)

destroyEnv :: Env -> IO ()
destroyEnv (Env (Client url sess) instanceID _ _) = do
    _ <- WreqS.post sess (url <> "/v1/envs/" <> instanceID <> "/close/") B.empty
    return ()

-- | Convenient combination of 'withClient' and 'withEnv'.
withEnv' :: String -> String -> (Env -> IO a) -> IO a
withEnv' url envID f = withClient url (\c -> withEnv c envID f)

-- | Specification of an action or observation space.
data Space = Discrete Int64
             -- ^ Integers in the range [0, n).
           | Box [Int64] [Float] [Float]
             -- ^ Multidimensional array of real numbers. The first list
             -- contains the size of each dimension and the other two lists
             -- contain the lower and upper bounds of each element respectively.
           deriving (Show)

-- | Sample a point from a 'Space' at uniform random.
sample :: Space -> IO (S.Vector Float)
sample (Discrete n) = S.singleton . fromIntegral <$> randomRIO (0, n - 1)
sample (Box _ low high) = S.fromList <$> zipWithM (curry randomRIO) low high

getSpace :: Client -> String -> String -> IO Space
getSpace (Client url sess) instanceID urlSuffix = do
    let reqURL = url <> "/v1/envs/" <> instanceID <> urlSuffix
    resp <- Wreq.asJSON =<< WreqS.get sess reqURL
    let body = resp ^. Wreq.responseBody :: Aeson.Value
        info = fromMaybe (error ("Bad space json: " <> show body))
                         (body ^? key "info" . _Value)
    return $ case info ^. key "name" . _String of
        "Discrete" ->
            case info ^? key "n" . _Integral of
                Nothing -> error ("Bad discrete space json: " <> show body)
                Just n -> Discrete n
        "Box" ->
            let shape = info ^.. key "shape" . values . _Integral
                low = info ^.. key "low" . values . _Float
                high = info ^.. key "high" . values . _Float
            in if null shape ||
                  fromIntegral (product shape) /= length low ||
                  fromIntegral (product shape) /= length high
                  then error ("Bad box space json: " <> show body)
                  else Box shape low high
        spaceName -> error $ "Unsupported space name: " <> Text.unpack spaceName

-- | Reset the environment and return the initial observation.
reset :: Env -> IO (S.Vector Float)
reset (Env (Client url sess) instanceID _ _) = do
    let reqURL = url <> "/v1/envs/" <> instanceID <> "/reset/"
    resp <- Wreq.asJSON =<< WreqS.post sess reqURL B.empty
    let body = resp ^. Wreq.responseBody :: Aeson.Value
    return (S.fromList (body ^.. key "observation" . values . _Float))

data StepResult = StepResult
    { stepObservation :: S.Vector Float
    , stepReward :: Float
    , stepDone :: Bool
    }

-- | Perform an action.
step :: Env
     -> S.Vector Float  -- ^ Action.
     -> Bool             -- ^ Whether to render the screen.
     -> IO StepResult
step env@(Env (Client url sess) instanceID _ _) action render = do
    let reqURL = url <> "/v1/envs/" <> instanceID <> "/step/"
        actionJSON = case actionSpace env of
            Discrete _ ->
                if S.length action /= 1
                    then error $ printf
                        "Bad action length for discrete space: \
                        \expected=1 got=%v"
                        (S.length action)
                    else Aeson.toJSON (truncate (S.head action) :: Int64)
            Box shape _ _ ->
                let expectedLen = fromIntegral (product shape)
                in if S.length action /= expectedLen
                    then error $ printf
                        "Bad action length for box space with shape %v: \
                        \expected=%v got=%v"
                        (show shape) expectedLen (S.length action)
                    else Aeson.toJSON action
        reqBody = Aeson.Object $ HMap.fromList
            [ ("action", actionJSON)
            , ("render", Aeson.Bool render)
            ]
    resp <- Wreq.asJSON =<< WreqS.post sess reqURL reqBody
    let body = resp ^. Wreq.responseBody :: Aeson.Value
    return StepResult
        { stepObservation =
              S.fromList (body ^.. key "observation" . values . _Float)
        , stepReward = fromMaybe 0 (body ^? key "reward" . _Float)
        , stepDone = fromMaybe False (body ^? key "done" . _Bool)
        }


_Float :: AsNumber t => Prism' t Float
_Float = _Number . iso Scientific.toRealFloat realToFrac
{-# INLINE _Float #-}
