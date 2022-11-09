#include <iostream>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <fstream>
#include <gflags/gflags.h>
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/lcm/drake_lcm.h"
#include <drake/systems/lcm/lcm_interface_system.h>
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/primitives/vector_log_sink.h"
#include "drake/common/eigen_types.h"

#include "utils.h"
#include "forceDisturber.h"
#include "plantIK.h"
#include "common_sys.h"
#include "traj.h"
#include "controller.h"
#include "hardware_plant.h"

DEFINE_double(dt, 2e-3, "Control period.");//控制周期
DEFINE_double(realtime, 1.0, "Target realtime rate.");
DEFINE_bool(log, false, "Record log");
DEFINE_bool(pub, true, "Publish lcm msg");
DEFINE_bool(real, false, "Run real");
DEFINE_uint32(traj, 0, "Soure traj");//标志位，运行程序可选
DEFINE_uint32(count, 1, "Jump count");

#define MAX_CURRENT (31.2)//最大电流

static const char *model_path = "models/singleleg_v3/urdf/singleleg_v3_symmetrical.urdf";//加载模型路径
static const char *kLcmStateChannel = "singleleg/state";
static const char *kLcmCmdChannel = "singleleg/commond";
static const std::string traj_file("data/singleleg_v3_qv_traj.csv");//单腿跳离线轨迹优化
static double com_squat = 0.175;//初始化下蹲的距离
static uint16_t nq_f = 7, nv_f = 6;//四元数 + xyz/   欧拉角速度 + xyz的线速度 
static Eigen::VectorXd q_initial;//初始化广义角度
static std::vector<std::string> initial_joint_name{"joint_knee"};//初始关节-> 膝关节
static std::vector<double> initial_joint_pos{0.4};
static std::vector<double> joint_effort_limits{1e3, 1e3, 1e3};
std::string model_instance_name("singleleg_v3");
static std::vector<std::string> contacts_frame_name{"foot_toe_l", "foot_toe_r", "foot_heel_l", "foot_heel_r"};//左右脚趾，左右脚尖

std::vector<std::string> end_frames_name{"base_link", "foot_sole"};//末端坐标系名，（基座，脚趾）
bool time_run = false;
bool traj_reset = false;
lcm::LCM lc;
Eigen::Vector3d foot_pos;//足端位置三维

namespace drake
{
  void initialState(multibody::MultibodyPlant<double> *plant, systems::Context<double> *plant_context,
                    double com_squat,//初始化下蹲的距离
                    std::vector<std::string> &end_frames_name,//基座和脚趾的坐标系
                    std::vector<std::string> &initial_joint_name, std::vector<double> &initial_joint_pos)//初始化膝关节高度0.4
  {
    math::RigidTransformd footInBase = plant->GetFrameByName(end_frames_name[1]).CalcPose(*plant_context, plant->GetFrameByName(end_frames_name[0]));
    Eigen::VectorXd q0 = plant->GetPositions(*plant_context);
    q0[6] = -footInBase.translation()[2];//广义坐标系下机身高度
    plant->SetPositions(plant_context, q0);
    Eigen::VectorXd r0 = plant->CalcCenterOfMassPositionInWorld(*plant_context);//世界系下质心的位置xyz

    for (uint32_t i = 0; i < initial_joint_name.size(); i++)//初始值i=0，.size()获取字符串长度为10
    {
      const multibody::RevoluteJoint<double> &joint = plant->GetJointByName<multibody::RevoluteJoint>(initial_joint_name[i]);
      joint.set_angle(plant_context, initial_joint_pos[i]);//设置膝关节，关节角度？
    }
    q0 = plant->GetPositions(*plant_context);//初始化广义角度

    if (FLAGS_traj == 0)
    {
      CoMIK ik(plant, end_frames_name, 1e-6, 1e-6);
      std::vector<std::vector<Eigen::Vector3d>> pose_vec{//这个数组的含义没太看懂？
          {Eigen::Vector3d(0, 0.349, 0), Eigen::Vector3d(0, r0[1], r0[2] - com_squat)},
          {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, footInBase.translation()[1], 0)}};
      Eigen::VectorXd q;
      bool result = ik.solve(pose_vec, q0, q);
      if (result == false)
      {
        throw std::runtime_error("Failed to IK0!");//异常必须显式地抛出，才能被检测和捕获到；
      }
      plant->SetPositions(plant_context, q);
      plant->SetVelocities(plant_context, Eigen::VectorXd::Constant(plant->num_velocities(), 0));
      q_initial = q;//赋值位置，速度，关节角度
    }
    else if (FLAGS_traj == 1)//标志位为1时，读取离线轨迹
    {
      uint32_t nq = plant->num_positions();//得到广义角度的个数
      std::vector<std::vector<double_t>> traj;
      if (!readCsvData(traj_file.c_str(), true, traj))//如果文件有误，则抛出错误
      {
        throw std::runtime_error("Failed to open " + std::string(traj_file));
      }
      Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(traj[0].data(), nq, 1);//为了构建Map变量，我们需要其余的两个信息：一个指向元素数组的指针，Matrix/vector的尺寸
      plant->SetPositions(plant_context, q);
      plant->SetVelocities(plant_context, Eigen::VectorXd::Constant(plant->num_velocities(), 0));
      q_initial = q;
      Eigen::Vector3d torso_rot = math::RollPitchYawd(Eigen::Quaterniond(q[0], q[1], q[2], q[3])).vector();//机身的四元数转旋转矩阵
      std::cout<<"init torso_rot: "<<torso_rot.transpose()*TO_DEGREE<<std::endl;
    }
  }
} // namespace drake

namespace drake
{
  void planPosePP(multibody::MultibodyPlant<double> *plant, systems::Context<double> *plant_context,
                  std::vector<std::string> &end_frames_name,
                  trajectories::PiecewisePolynomial<double> &e_pp)//分段多项式
  {
    math::RigidTransformd foot_in_base = plant->GetFrameByName(end_frames_name[1]).CalcPose(*plant_context, plant->GetFrameByName(end_frames_name[0]));//机身到脚掌的距离
    double separate_half = foot_in_base.translation()[1];//机身与脚掌的y距离
    Eigen::VectorXd r0 = plant->CalcCenterOfMassPositionInWorld(*plant_context);//世界系下的质心位置

    double A = 0.1, T = 3;
    std::vector<Eigen::Vector3d> com{
        Eigen::Vector3d(r0[0], r0[1], r0[2]),
        Eigen::Vector3d(r0[0], r0[1], r0[2]),
        Eigen::Vector3d(r0[0], r0[1], r0[2] - A),
        Eigen::Vector3d(r0[0], r0[1], r0[2]),
        Eigen::Vector3d(r0[0], r0[1], r0[2] - A),
        Eigen::Vector3d(r0[0], r0[1], r0[2]),
        Eigen::Vector3d(r0[0], r0[1], r0[2])};
    std::vector<Eigen::Vector3d> base_rot{
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0)};
    std::vector<Eigen::Vector3d> lfoot_rot{
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0)};
    std::vector<Eigen::Vector3d> lfoot_pos{
        Eigen::Vector3d(0, separate_half, 0),
        Eigen::Vector3d(0, separate_half, 0),
        Eigen::Vector3d(0, separate_half, 0),
        Eigen::Vector3d(0, separate_half, 0),
        Eigen::Vector3d(0, separate_half, 0),
        Eigen::Vector3d(0, separate_half, 0),
        Eigen::Vector3d(0, separate_half, 0)};
    std::vector<Eigen::Vector3d> rfoot_rot{
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0, 0, 0)};
    std::vector<Eigen::Vector3d> rfoot_pos{
        Eigen::Vector3d(0, -separate_half, 0),
        Eigen::Vector3d(0, -separate_half, 0),
        Eigen::Vector3d(0, -separate_half, 0),
        Eigen::Vector3d(0, -separate_half, 0),
        Eigen::Vector3d(0, -separate_half, 0),
        Eigen::Vector3d(0, -separate_half, 0),
        Eigen::Vector3d(0, -separate_half, 0)};//将机身和脚掌的状态分成7段

    std::vector<double> times;
    std::vector<Eigen::MatrixXd> knots;
    double t = 0, dt = T / 6.0;
    for (uint32_t i = 0; i < com.size(); i++)//com.size()猜测是7
    {
      times.push_back(t);//函数将一个新的元素加到vector的最后面，位置为当前最后一个元素的下一个元素
      t += dt;
      Eigen::VectorXd knot(18);
      knot << base_rot[i], com[i], lfoot_rot[i], lfoot_pos[i], rfoot_rot[i], rfoot_pos[i];//3*6
      knots.push_back(knot);//每次在knots尾部再加18个元素
    }
    e_pp = trajectories::PiecewisePolynomial<double>::CubicShapePreserving(times, knots, true);//分段多项式插值
  }
} // namespace drake

namespace drake
{
  class Trajectory : public systems::LeafSystem<double>
  {
  public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Trajectory)

    Trajectory(multibody::MultibodyPlant<double> *plant, std::vector<std::string> end_frames_name,
               uint32_t nq_f = 7, uint32_t nv_f = 6)
        : plant_(plant),
          end_frames_name_(end_frames_name),
          nq_f_(nq_f),
          nv_f_(nv_f)//把括号里面的赋值给外面的
    {
      plant_context_ = plant_->CreateDefaultContext();//创建默认容器
      na_ = plant_->num_actuated_dofs();//获取驱动关节的数目 3
      nq_ = plant_->num_positions();//获取广义位置的数目，包括四元数，质心3个位置以及3个关节角度
      nv_ = plant_->num_velocities();//获取广义速度的数目，包括RPY角速度，线速度，关节角速度
      dt_ = plant_->time_step();//获取时间步长
      gravity_vec_ = plant_->gravity_field().gravity_vector();//获取重力向量
      robot_total_mass_ = plant_->CalcTotalMass(*plant_context_);//获取机器人总质量

      DeclareAbstractInputPort("state_ext", Value<EstimateState_t>{});
      DeclareAbstractOutputPort("state_des", &Trajectory::Output);
      DeclareVectorOutputPort("desire", nq_ + nv_, &Trajectory::OutputDesire);
      DeclarePeriodicDiscreteUpdateEvent(dt_, 0, &Trajectory::Update);//设置了输入输出接口

      state_des_.phase = double_stand;//初始化相位设置为double_stand
      state_des_.q.resize(nq_);//resize()，设置大小（size）;
      state_des_.v.resize(nv_);
      state_des_.vd.resize(nv_);
      prev_state_des_.phase = double_stand;
      prev_state_des_.q.resize(nq_);
      prev_state_des_.v.resize(nv_);
      prev_state_des_.vd.resize(nv_);
      comv_ik_ = std::make_shared<CoMVIK>(plant_, end_frames_name_);//官方鼓励用make_shared函数来创建对象，而不要手动去new，这样就可以防止我们去使用原始指针创建多个引用计数体系。
      state_takeoff_.q.resize(nq_);
      state_takeoff_.v.resize(nv_);
      state_takeoff_.vd.resize(nv_);//以上设置变量的尺寸大小
    }

    const systems::InputPort<double> &get_state_est_input_port() const
    {
      return systems::LeafSystem<double>::get_input_port(0);
    }

    const systems::OutputPort<double> &get_state_des_output_port() const
    {
      return systems::LeafSystem<double>::get_output_port(0);
    }

    const systems::OutputPort<double> &get_desire_output_port() const
    {
      return systems::LeafSystem<double>::get_output_port(1);
    }//接口传输数据

    void Initialize(systems::Context<double> &context, const Eigen::VectorXd q0) const//类的构造函数的初始化
    {
      Eigen::VectorXd qv(nq_ + nv_);//10+9
      qv << q0, Eigen::VectorXd::Constant(nv_, 0);//Eigen::VectorXd::Constant(nv_, 0)，定义9维常量列向量，用0填充元素值
      plant_->SetPositionsAndVelocities(plant_context_.get(), qv);//把qv取回，塞到plant_里面
      Eigen::Vector3d r0 = plant_->CalcCenterOfMassPositionInWorld(*plant_context_.get());//计算质心位置

      state_des_.q = qv.segment(0, nq_);//把qv的前10个值赋值给前面q
      state_des_.v = qv.segment(nq_, nv_);//把qv的后9个值赋值给前面v
      state_des_.vd = Eigen::VectorXd::Constant(nv_, 0); //加速度定义为全0的n_v维的常数列向量
      UpdateStateWithQV(state_des_);//更新q,qv,vd
      prev_state_des_ = state_des_;//积分用
      state_des_qfb_ = state_des_;//髋关节补偿用，让质心位置和脚掌位置在一条直线上，保持高度

      torso_rot_des_ << 0, 20 * (M_PI / 180.0), 0;//躯干角度
      rdd_cmd_ << 0, 0, 0;//质心加速度
      rd_takeoff_des_ << 0, 0, 0;//起跳速度
      theta_des_ = rd_takeoff_des_[0] * 0.2;
      lu_ = r0[2] + 0.05;
      ll_ = r0[2];
      l_w_ = 0;//L中theata的角速度
      l_w_integral_ = 0;
      flags = 0;
    }

  private://为了防止某些数据成员或成员函数从外部被直接访问，可以将它们声明为private，这样编译器会阻止任何来自外部非友元的直接访问。
    void UpdateStateWithQV(EstimateState_t &state) const//更新实际的所有state，q，v
    {
      Eigen::VectorXd qv(nq_ + nv_);
      qv << state.q, state.v;
      plant_->SetPositionsAndVelocities(plant_context_.get(), qv);//把qv取回，塞到plant_里面

      state.r = plant_->CalcCenterOfMassPositionInWorld(*plant_context_);
      state.rd = plant_->CalcCenterOfMassTranslationalVelocityInWorld(*plant_context_);
      state.rdd = plant_->CalcBiasCenterOfMassTranslationalAcceleration(*plant_context_, multibody::JacobianWrtVariable::kV,
                                                                        plant_->world_frame(),
                                                                        plant_->world_frame());

      math::RigidTransformd foot_in_w;
      multibody::SpatialVelocity<double> footv_in_w;//相对机身刚体的脚掌的[角度，位置]
      foot_in_w = plant_->GetFrameByName(end_frames_name_[1]).CalcPose(*plant_context_.get(), plant_->world_frame());//相对机身刚体的脚掌的[角度，位置]
      footv_in_w = plant_->GetFrameByName(end_frames_name_[1]).CalcSpatialVelocityInWorld(*plant_context_.get());//相对机身刚体的脚掌的[角速度，速度]
      state.lf << math::RollPitchYawd(foot_in_w.rotation()).vector(), foot_in_w.translation();//相对机身刚体的脚掌的[角度，位置]
      state.lfv = footv_in_w.get_coeffs();//相对机身刚体的脚掌的[角速度，速度]

      multibody::SpatialMomentum<double> L_WScm_W;//空间动量
      L_WScm_W = plant_->CalcSpatialMomentumInWorldAboutPoint(*plant_context_, state.r);
      state.cm << L_WScm_W.rotational(), L_WScm_W.translational();//质心动量，角动量+线动量
      L_WScm_W = plant_->CalcSpatialMomentumInWorldAboutPoint(*plant_context_, state.lf.segment(3, 3));
      state.lf_sm << L_WScm_W.rotational(), L_WScm_W.translational();//脚掌的动量
    }

    void UpdateStatePositionWithQV(EstimateState_t &state) const//更新qv中实际末端位置
    {
      Eigen::VectorXd qv(nq_ + nv_);
      qv << state.q, state.v;
      plant_->SetPositionsAndVelocities(plant_context_.get(), qv);

      state.r = plant_->CalcCenterOfMassPositionInWorld(*plant_context_);

      math::RigidTransformd foot_in_w;
      multibody::SpatialVelocity<double> footv_in_w;
      foot_in_w = plant_->GetFrameByName(end_frames_name_[1]).CalcPose(*plant_context_.get(), plant_->world_frame());
      state.lf << math::RollPitchYawd(foot_in_w.rotation()).vector(), foot_in_w.translation();
    }

    void Update(const systems::Context<double> &context, systems::DiscreteValues<double> *next_state) const
    {
      static struct timespec t0, t1;
      struct timespec t0_sol, t1_sol;

      // clock_gettime(CLOCK_MONOTONIC, &t0);
      const auto &state_est = get_state_est_input_port().Eval<EstimateState_t>(context);

      if (kbhit())
      {
        char ch = getchar();
        if (ch == 's')
        {
          if (!time_run)//time_run默认是flase
          {
            std::cout << "Take off.\n";
            // time_run = true;
            state_des_.phase = take_off;
            state_des_.phase_time = 0;
          }
        }
        // if (ch == 'd')
        // {
        //   std::cout << "Take off.\n";
        //   time_run = false;
        // }
      }
      Eigen::Vector3d torso_rot = math::RollPitchYawd(Eigen::Quaterniond(state_est.q[0], state_est.q[1], state_est.q[2], state_est.q[3])).vector();//躯干实际的欧拉角
      Eigen::Vector3d lv = state_est.r - state_est.lf.segment(3, 3);//质心减去脚掌的向量L,脚掌后三个元素为位置量
      double l = lv.norm();//norm返回的是向量的二范数
      double theta = atan((state_est.r[0] - state_est.lf.segment(3, 3)[0]) / (state_est.r[2] - state_est.lf.segment(3, 3)[2]));//与垂线的夹角
      double theta_ik = atan((state_des_.r[0] - state_des_.lf.segment(3, 3)[0]) / (state_des_.r[2] - state_des_.lf.segment(3, 3)[2]));//逆解出来的与垂线的夹角
      Eigen::Vector3d torso_rot_ik = math::RollPitchYawd(Eigen::Quaterniond(state_des_.q[0], state_des_.q[1], state_des_.q[2], state_des_.q[3])).vector();//躯干期望的欧拉角
      Eigen::Vector3d lv_ik = state_des_.r - state_des_.lf.segment(3, 3);//期望的L向量

      if (state_des_.phase == take_off)
      {
        state_des_.cm.setZero();

        V_takeoff_ = 1.5;
        double S_takeoff = 0.08;
        double a_takeoff = 0;
        double t_takeoff = 0;
        a_takeoff = V_takeoff_ * V_takeoff_/(2 * S_takeoff);
        t_takeoff = 2 * S_takeoff/V_takeoff_;
        state_des_.rdd << 0, 0, a_takeoff;
        state_des_.rd = prev_state_des_.rd + state_des_.rdd * dt_;
        state_des_.lfv.setZero();//起跳阶段了控制了质心的加速度，速度以及脚掌的速度置为0
         
        if (state_des_.phase_time > t_takeoff)
        {
          {
            std::cout << "Flight.\n";
            state_des_.phase = flight;
            state_des_.phase_time = 0;
          }
        }
      }
      else if (state_des_.phase == flight)
      { 
        double V_flight = 0;
        double a_flight = -9.81;
        double a_f_flight = 0;
        double t_Vtogether = 0.;
        double V_Vtogether = 0.;
        double S_together_flight = 0.02;
        
        t_Vtogether = 2 * S_together_flight/V_takeoff_;
        V_Vtogether = V_takeoff_ + a_flight * t_Vtogether;
        a_f_flight = V_Vtogether/t_Vtogether;
        
        state_des_.rdd << 0, 0, a_flight;
        state_des_.rd = prev_state_des_.rd + state_des_.rdd * dt_;
        state_des_.lfv << 0, 0, 0, 0, 0, a_f_flight * state_des_.phase_time;//腾空阶段共速前控制了质心的加速度，速度以及脚掌的速度为匀加

        if (state_des_.phase_time > t_Vtogether)
        {
          state_des_.lfv.segment(0, 3) = (Eigen::Vector3d(0, 0, 0) - state_est.lf.segment(0, 3)) * 10;//腾空共速后，需要对脚掌姿态进行调控
          Eigen::Vector3d w = Eigen::Vector3d(0, (theta_des_ - theta) * 10, 0);
          l_w_ = w[1];//通过Y轴的角速度来控制L的pitch角
          if(l_w_integral_ > 10*TO_RADIAN || l_w_integral_ < -20*TO_RADIAN)//TO_RADIAN转化为弧度制
          {
             w[1] = 0;
             l_w_ = 0;//限幅值
          }
          l_w_integral_ += l_w_*dt_;//积分得到L的pitch角
          Eigen::Vector3d lfv_lvec = (Eigen::Vector3d(0, 0, ll_) - lv) * -1;//补偿值
          state_des_.lfv.segment(3, 3) = state_des_.rd + lv.cross(w) + lfv_lvec;//V=V0+LxW+补偿值
        }
        Eigen::Vector3d torso_rot_des = math::RollPitchYawd(Eigen::Quaterniond(state_des_.q[0], state_des_.q[1], state_des_.q[2], state_des_.q[3])).vector();//期望的躯干欧拉角
        torso_rot_des = 0.9*torso_rot_des + 0.1*torso_rot;//类似滤波作用，减小突变
        auto torso_Q = math::RollPitchYawd(torso_rot_des).ToQuaternion();//躯干四元数
        state_des_.q.segment(0,4)<<torso_Q.w(),torso_Q.x(),torso_Q.y(),torso_Q.z();
        state_des_.q.segment(4,3) = 0.9*state_des_.q.segment(4,3) + 0.1*state_est.q.segment(4,3);//类似滤波作用，减小突变

        if (state_est.phase == touch_down)
        {
            std::cout << "Touch down.\n";
            state_des_.phase = touch_down;
            state_des_.phase_time = 0;

            state_des_.rd = state_est.rd;
            state_des_.r = state_est.r;
            state_des_.rd[0] = 0;
            state_des_.lfv.setZero();
            V_touchdown_ = state_est.rd[2];//记录触地时刻的速度
        }
      }
      else if (state_des_.phase == touch_down)
      {
        double a_touchdown = 0;
        double t_touchdown = 0;
        a_touchdown = V_touchdown_ * V_touchdown_ /(2*(0.1));//加速距离与起跳阶段加速距离保持一致，0.1
        t_touchdown = -V_touchdown_/a_touchdown;
        state_des_.rdd << 0, 0, a_touchdown;
        state_des_.rd = prev_state_des_.rd + state_des_.rdd * dt_;
        state_des_.rd[0] = 10 * (theta_des_ - theta_ik);//通过L的theta来控制质心X的速度

        // foot_touch = state_des_.lf.segment(3,3);
        // state_des_.rd[0] += 1.0 * (theta_des_ - theta) + 0.4 * (0 - state_est.rd[0]);//x feedback controller
        // state_des_.rd[0] = LIMITING(state_des_.rd[0], -0.4, 0.4);

        // double rdd2_fb = 1.0 * (lv_ik[2] - lv[2]) + 0.4 * (state_des_.rd[2] - state_est.rd[2]);
        // rdd2_fb = LIMITING( rdd2_fb, -0.4, 0.4);
        // state_des_.rdd[2] = state_des_.rdd[2] + rdd2_fb;

        state_des_.lfv.setZero();//落地足端速度置0
        state_des_.lfv[1] = 10*(theta_ik-state_des_.lf[1]); //TODO: assume theta_des_=0
        //touchdown阶段通过theta_ik分别调控质心水平位置和脚掌的水平位置
        if (state_des_.phase_time > t_touchdown)
        {
          n_jump_ = n_jump_ + 1;//记录跳跃次数
          {
            std::cout << "n_jump = "<<n_jump_<<std::endl;
            state_des_.phase = double_stand;
            state_des_.phase_time = 0;

            state_des_.rdd << 0, 0, 0;
            state_des_.rd << 0, 0, 0;
            state_des_.r = prev_state_des_.r;
          }
        }
      }
      else if (state_des_.phase == double_stand)
      {
        state_des_.cm[1] = (torso_rot_des_[1] - torso_rot_ik[1]) * 1;
        state_des_.cm[1] = LIMITING(state_des_.cm[1], -0.2, 0.2);

        state_des_.rdd << 0, 0, 0;
        state_des_.rd << 0, 0, 0;
        state_des_.rd[0] = 10 * (theta_des_ - theta_ik);//控制质心水平位置
        state_des_.rd[2] = 10 * (ll_ - lv_ik[2]);//控制质心高度
        state_des_.rd[2] = LIMITING(state_des_.rd[2], -0.2, 0.2);//将质心Z速度限幅

        state_des_.lfv.setZero();
        state_des_.lfv[1] = 10*(theta_ik-state_des_.lf[1]);//只调控绕Y的速度
      }
      state_des_.phase_time += dt_;

      state_des_.cm.segment(0, 3)[0] = 0;
      state_des_.cm.segment(0, 3)[2] = 0;//将期望的质心动量，X轴和Z轴的角动量设置为0
      state_des_.rd[1] = 0;//质心的Y速度置为0
      state_des_.lfv.segment(0, 3)[0] = 0;
      state_des_.lfv.segment(0, 3)[2] = 0;
      state_des_.lfv.segment(3, 3)[1] = 0;//只保留脚掌的Y轴角速度

      prev_state_des_ = state_des_; // no assign with prev_state_des_ except here//除了这里，prev_state_des_没有赋值
      std::vector<std::vector<Eigen::Vector3d>> pose_vec{
          {state_des_.cm.segment(0, 3) / robot_total_mass_, state_des_.rd},
          {state_des_.lfv.segment(0, 3), state_des_.lfv.segment(3, 3)}};//姿态向量包括（机身角速度、机身线速度；脚掌角速度、脚掌线速度）
      bool result = comv_ik_->solve(pose_vec, prev_state_des_.q, prev_state_des_.v, dt_, state_des_.q, state_des_.v);//基于速度的逆解
      if (result == false)
      {
        std::cout << "torso: " << pose_vec[0][0].transpose() << ", " << pose_vec[0][1].transpose() << "\n";
        std::cout << "lfoot: " << pose_vec[1][0].transpose() << ", " << pose_vec[1][1].transpose() << "\n";
        throw std::runtime_error(std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", Failed to IK!");//抛出逆解错误
      }
      state_des_.vd = (state_des_.v - prev_state_des_.v) / dt_;//计算期望的加速度vd，6维
      for (uint32_t i = 0; i < nv_; i++)
      {
        state_des_.vd[i] = LIMITING(state_des_.vd[i], -2e3, 2e3);//对计算的期望的加速度vd，6维进行限幅
      }
      UpdateStatePositionWithQV(state_des_); // update endpoint position according to qv  只更新末端位置

      state_des_qfb_ = state_des_;//将期望值赋值给补偿用
      if(state_des_.phase == touch_down || state_des_.phase == double_stand)//在着地的时候进行调控质心位置
      {
        double hippitch_offset = -1.*(state_des_.q[nq_f_ + 1] - state_est.q[nq_f_ + 1]);//将膝关节的角度变化映射至髋关节补偿角度量
        hippitch_offset = LIMITING(hippitch_offset, -15 * TO_RADIAN, 15 * TO_RADIAN);//限幅+-15弧度
        state_des_qfb_.q[nq_f_ + 0] -= hippitch_offset;

        double q_ankle = 0.2*(state_des_.v[1]-state_est.v[1]);//将质心的pitch角速度映射到膝关节角度补偿量上
        q_ankle = LIMITING(q_ankle, -3 * TO_RADIAN, 3 * TO_RADIAN);//限幅+-3弧度
        state_des_qfb_.q[nq_f_ + 2] -= q_ankle;
        // std::cout << "q_ankle: " << q_ankle * TO_DEGREE<<" | "<<state_des_.v[1]<<" | "<<state_est.v[1]<< "\n";
      }

      clock_gettime(CLOCK_MONOTONIC, &t1);//clock_gettime（获取指定时钟的时间值），CLOCK_MONOTONIC:从系统启动这一刻起开始计时,不受系统时间被用户改变的影响。
#if 0
      printf("Period(ms): %f, Solve(ms): %f\n",
             TIME_DIFF(t0, t1) * 1000.0,
             TIME_DIFF(t0_sol, t1_sol) * 1000.0);
#endif

      if (FLAGS_pub)//发布消息
      {
        lcmPublishState(&lc, "desire", state_des_.q, state_des_.v, state_des_.vd);
        Eigen::Matrix<double, 6, 1> x_com_des;
        x_com_des << state_des_.r, state_des_.rd;//输出质心期望位置和速度
        // Eigen::Vector3d u_com_des(state_des_.rdd);
        Eigen::Vector3d u_com_des;
        // u_com_des<<0, 0, TIME_DIFF(t0, t1) * 1000.0;
        u_com_des<<theta_des_, theta, TIME_DIFF(t0, t1) * 1000.0;//输出期望theta角，实际theta，发布频率
        if(u_com_des[2]>1000)
          u_com_des[2] = 0;//过滤错误延时性数据
        lcmPublishVector(&lc, "desire/com/x", x_com_des);
        lcmPublishVector(&lc, "desire/com/u", u_com_des);
        lcmPublishVector(&lc, "desire/lfoot", state_des_.lf);
        lcmPublishVector(&lc, "desire/lfootv", state_des_.lfv);
        lcmPublishVector(&lc, "desire/L/com", state_des_.cm);
        lcmPublishVector(&lc, "desire/L/lfoot", state_des_.lf_sm);
        lcmPublishValue(&lc, "desire/phase", state_des_.phase);
      }
      clock_gettime(CLOCK_MONOTONIC, &t0);//获取指定时钟的时间值

    }

    void Output(const systems::Context<double> &context, EstimateState_t *output) const
    {
      (*output) = state_des_qfb_;//髋关节，膝关节角度补偿量
    }

    void OutputDesire(const systems::Context<double> &context, systems::BasicVector<double> *output) const
    {
      Eigen::VectorXd desire(nq_ + nv_);
      desire << state_des_.q, state_des_.v;
      output->SetFromVector(desire);
    }
    //以下为Traj类下的变量定义
    multibody::MultibodyPlant<double> *plant_;
    std::unique_ptr<systems::Context<double>> plant_context_;
    int32_t na_;
    int32_t nq_;
    int32_t nv_;
    uint32_t nq_f_;
    uint32_t nv_f_;
    double dt_;
    mutable std::vector<std::string> end_frames_name_;
    Eigen::Vector3d gravity_vec_;
    double robot_total_mass_;
    std::shared_ptr<CoMVIK> comv_ik_;

    mutable trajectories::PiecewisePolynomial<double> q_pp_, v_pp_, vd_pp_;
    mutable double time_{0};
    mutable EstimateState_t state_des_, prev_state_des_, state_des_qfb_;
    mutable EstimateState_t state_takeoff_;
    ;

    mutable Eigen::Vector3d foot_touch;
    mutable Eigen::Vector3d torso_rot_des_;
    mutable Eigen::Vector3d rdd_cmd_;
    mutable Eigen::Vector3d rd_takeoff_des_;
    mutable double l_w_;
    mutable double l_w_integral_;
    mutable double theta_des_;
    mutable double ll_, lu_;
    mutable uint8_t flags;
    mutable uint32_t count_jump_{0};
    mutable double V_takeoff_;
    mutable double V_touchdown_ ;
    mutable double S_touchdown_ ;
    mutable double t_flight_peirod_ ;
    mutable double V_takeoff_true_ ;
    mutable double n_jump_ = 0;
  };
} // namespace drake
//至此Traj类结束
namespace drake
{
  class StateExt : public systems::LeafSystem<double>
  {
  public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(StateExt)//删除了用于复制构造、复制分配、移动构造和移动分配的特殊成员函数。

    StateExt(multibody::MultibodyPlant<double> *plant, std::vector<std::string> end_frames_name)
        : plant_{plant},
          end_frames_name_{end_frames_name}
    {
      plant_context_ = plant_->CreateDefaultContext();//创建默认容器
      na_ = plant_->num_actuated_dofs();
      nq_ = plant_->num_positions();
      nv_ = plant_->num_velocities();
      dt_ = plant_->time_step();
      DeclareVectorInputPort("state", nq_ + nv_);//10+9
      DeclareAbstractOutputPort("state_ext", &StateExt::Output);
      DeclarePeriodicDiscreteUpdateEvent(dt_, 0, &StateExt::Update);

      prev_v_.resize(nv_);
      prev_v_.setZero();
      state_est_.phase = take_off;
      state_est_.q.resize(nq_);
      state_est_.v.resize(nv_);
      state_est_.vd.resize(nv_);
      state_est_.v.setZero();//设置真实值的尺寸，并给0
    }

    const systems::InputPort<double> &get_state_input_port() const
    {
      return systems::LeafSystem<double>::get_input_port(0);
    }

    const systems::OutputPort<double> &get_state_est_output_port() const
    {
      return systems::LeafSystem<double>::get_output_port(0);
    }

    void Initialize(const Eigen::VectorXd q0)
    {
      Eigen::VectorXd qv(nq_ + nv_);//10+9
      qv << q0, Eigen::VectorXd::Constant(nv_, 0);//初始化时，nq_赋值q0，将nv_全给0
      plant_->SetPositionsAndVelocities(plant_context_.get(), qv);//将qv塞给plant_

      prev_v_ = state_est_.v;
      state_est_.q = qv.segment(0, nq_);
      state_est_.v = qv.segment(nq_, nv_);
      state_est_.vd = (state_est_.v - prev_v_) / dt_;

      UpdateStateWithQV(state_est_);//更新实际值
    }

  private:
    void UpdateStateWithQV(EstimateState_t &state) const
    {
      Eigen::VectorXd qv(nq_ + nv_);
      qv << state.q, state.v;
      plant_->SetPositionsAndVelocities(plant_context_.get(), qv);

      state.r = plant_->CalcCenterOfMassPositionInWorld(*plant_context_);
      state.rd = plant_->CalcCenterOfMassTranslationalVelocityInWorld(*plant_context_);
      state.rdd = plant_->CalcBiasCenterOfMassTranslationalAcceleration(*plant_context_, multibody::JacobianWrtVariable::kV,
                                                                        plant_->world_frame(),
                                                                        plant_->world_frame());

      math::RigidTransformd foot_in_w;
      multibody::SpatialVelocity<double> footv_in_w;
      foot_in_w = plant_->GetFrameByName(end_frames_name_[1]).CalcPose(*plant_context_.get(), plant_->world_frame());//脚掌在世界系下的位置
      footv_in_w = plant_->GetFrameByName(end_frames_name_[1]).CalcSpatialVelocityInWorld(*plant_context_.get());//脚掌在世界系下的空间速度
      state.lf << math::RollPitchYawd(foot_in_w.rotation()).vector(), foot_in_w.translation();
      state.lfv = footv_in_w.get_coeffs();

      multibody::SpatialMomentum<double> L_WScm_W;//空间动量
      L_WScm_W = plant_->CalcSpatialMomentumInWorldAboutPoint(*plant_context_, state.r);
      state.cm << L_WScm_W.rotational(), L_WScm_W.translational();//质心角动量+线动量
      L_WScm_W = plant_->CalcSpatialMomentumInWorldAboutPoint(*plant_context_, state.lf.segment(3, 3));
      state.lf_sm << L_WScm_W.rotational(), L_WScm_W.translational();//脚掌角动量+线动量

      math::RigidTransformd transform;
      if (state.phase == take_off)
      {
        if (state.phase_time > 0.1)
        {
          transform = plant_->GetFrameByName(end_frames_name_[1]).CalcPose(*plant_context_, plant_->world_frame());//脚掌到世界系下的位置
          if (transform.translation()[2] > 1.0e-3)//判断脚掌离地
          {
            state.phase = flight;//切换这flight
            state.phase_time = 0;
          }
        }
      }
      else if (state.phase == flight)
      {
        if (state.phase_time > 0.1)
        {
          transform = plant_->GetFrameByName(end_frames_name_[1]).CalcPose(*plant_context_, plant_->world_frame());
          if (transform.translation()[2] < 5.0e-4)//判断脚掌触地
          {
            state.phase = touch_down;//切换这touchdowm
            state.phase_time = 0;
          }
        }
      }
      else if (state.phase == touch_down)
      {
        if (state.phase_time > 0.1)
        {
          state.phase = take_off;//目前期望的相位规划是double stand
          state.phase_time = 0;
        }
      }//仿真里现在处理比较简陋，之后打算的做法是仿真里读取关节扭矩，计算当前接触力，再用来判断触地的相位，这种方式实物和仿真可以统一起来
      state.phase_time += dt_;
    }
    //以下是时刻更新的量
    void Update(const systems::Context<double> &context, systems::DiscreteValues<double> *next_state) const
    {
      const auto &qv = get_state_input_port().Eval(context);
      prev_v_ = state_est_.v;
      state_est_.q = qv.segment(0, nq_);
      state_est_.v = qv.segment(nq_, nv_);
      state_est_.vd = (state_est_.v - prev_v_) / dt_;

      UpdateStateWithQV(state_est_);

      if (FLAGS_pub)
      {
        lcmPublishState(&lc, "state", state_est_.q, state_est_.v, state_est_.vd);
        Eigen::Matrix<double, 6, 1> x_com;
        x_com << state_est_.r, state_est_.rd;
        lcmPublishVector(&lc, "state/com/x", x_com);
        lcmPublishVector(&lc, "state/lfoot", state_est_.lf);
        lcmPublishVector(&lc, "state/lfootv", state_est_.lfv);
        lcmPublishVector(&lc, "state/L/com", state_est_.cm);
        lcmPublishVector(&lc, "state/L/lfoot", state_est_.lf_sm);
        lcmPublishValue(&lc, "state/phase", state_est_.phase);
      }
    }

    void Output(const systems::Context<double> &context, EstimateState_t *output) const
    {
      (*output) = state_est_;//将真实的状态当作输出
    }

    multibody::MultibodyPlant<double> *plant_;
    std::unique_ptr<systems::Context<double>> plant_context_;
    int32_t na_;
    int32_t nq_;
    int32_t nv_;
    double dt_;
    std::vector<std::string> end_frames_name_;
    mutable Eigen::VectorXd q_, v_, vdot_;
    mutable Eigen::VectorXd prev_v_;
    mutable double time_{0};
    mutable EstimateState_t state_est_;
  };
} // namespace drake

namespace drake
{
  class ActuationMultiplexer : public systems::LeafSystem<double>//驱动多路转接器
  {
  public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ActuationMultiplexer)

    ActuationMultiplexer(multibody::MultibodyPlant<double> *plant, uint32_t nq_f = 7, uint32_t nv_f = 6)
        : plant_(plant),
          nq_f_(nq_f),
          nv_f_(nv_f)
    {
      plant_context_ = plant_->CreateDefaultContext();//创建默认容器
      na_ = plant_->num_actuated_dofs();
      nq_ = plant_->num_positions();
      nv_ = plant_->num_velocities();
      DeclareVectorInputPort("tau", na_);
      DeclareVectorInputPort("tau_max", na_);
      DeclareVectorOutputPort("output", na_ * 2, &ActuationMultiplexer::Output);
    }

    const systems::InputPort<double> &get_tau_input_port() const
    {
      return systems::LeafSystem<double>::get_input_port(0);//"tau", na_
    }

    const systems::InputPort<double> &get_tau_max_input_port() const
    {
      return systems::LeafSystem<double>::get_input_port(1);//"tau_max", na_
    }

  private:
    void Output(const systems::Context<double> &context, systems::BasicVector<double> *output) const
    {
      const auto &tau = get_tau_input_port().Eval(context);
      const auto &tau_max = get_tau_max_input_port().Eval(context);
      Eigen::VectorXd cmd(na_ * 2);
      cmd << tau, tau_max;//扭矩，最大扭矩
      output->SetFromVector(cmd);
    }

    multibody::MultibodyPlant<double> *plant_;
    std::unique_ptr<systems::Context<double>> plant_context_;//unique_ptr智能指针在容器中使用
    uint32_t nq_;
    uint32_t nv_;
    uint32_t na_;
    uint32_t nq_f_;
    uint32_t nv_f_;
  };
} // namespace drake

namespace drake
{//记录日志
  void recordLog(multibody::MultibodyPlant<double> *plant, const drake::systems::VectorLog<double> &log)
  {
    const char *fileName = "./data/state.csv";
    std::ofstream fp;
    fp.open(fileName);
    if (!fp.is_open())
    {
      throw std::runtime_error("Failed to open " + std::string(fileName));
    }

    fp << "time, ";
    for (uint32_t i = 0; i < plant->num_positions(); i++)
    {
      fp << "q" << i << ", ";
    }
    for (uint32_t i = 0; i < plant->num_velocities() - 1; i++)
    {
      fp << "v" << i << ", ";
    }
    fp << "v" << plant->num_velocities() - 1 << "\n";

    std::cout << "Write data ...\n";
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    const int m = log.num_samples();
    const int n = log.data().rows() + 1;
    Eigen::MatrixXd data(m, n);
    data << log.sample_times(), log.data().transpose();
    fp << data.format(CSVFormat);
    std::cout << "Write complete\n";
    fp.close();
  }

  int doMain()//建立逻辑框图，仿真or实物设置
  {
    systems::DiagramBuilder<double> builder;
    multibody::MultibodyPlant<double> *plant = builder.AddSystem<multibody::MultibodyPlant>(FLAGS_dt);
    if (!FLAGS_real)//运行仿真
    {
      systems::lcm::LcmInterfaceSystem *lcm = builder.AddSystem<systems::lcm::LcmInterfaceSystem>();
      geometry::SceneGraph<double> *scene_graph = builder.AddSystem<geometry::SceneGraph>();
      plant->RegisterAsSourceForSceneGraph(scene_graph);
      builder.Connect(plant->get_geometry_poses_output_port(), scene_graph->get_source_pose_port(plant->get_source_id().value()));//get_geometry_poses_output_port发出给get_source_pose_port
      builder.Connect(scene_graph->get_query_output_port(), plant->get_geometry_query_input_port());
      geometry::DrakeVisualizerd::AddToBuilder(&builder, *scene_graph, lcm);

      const multibody::ModelInstanceIndex model_instance = multibody::Parser(plant, scene_graph).AddModelFromFile(model_path);
      // Add model of the ground.
      const double static_friction = 1.0;//静摩擦力
      const Vector4<double> color(0.9, 0.9, 0.9, 1.0);//颜色设置
      plant->RegisterVisualGeometry(plant->world_body(),
                                    math::RigidTransformd(), geometry::HalfSpace(),
                                    "GroundVisualGeometry",
                                    color);
      // For a time-stepping model only static friction is used.
      const multibody::CoulombFriction<double> ground_friction(static_friction, static_friction);
      plant->RegisterCollisionGeometry(plant->world_body(),
                                       math::RigidTransformd(),
                                       geometry::HalfSpace(),
                                       "GroundCollisionGeometry",
                                       ground_friction);
    }
    else//连接实物
    {
      multibody::Parser(plant).AddModelFromFile(model_path);
    }
    plant->Finalize();

    StateExt *state_ext;
    Trajectory *trajectory;
    HardwarePlant *hard_plant;
    ActuationMultiplexer *act_mul;//指针赋值，右边赋值给左边
    trajectory = builder.AddSystem<Trajectory>(plant, end_frames_name);

    uint32_t nv = plant->num_velocities();
    uint32_t na = plant->num_actuated_dofs();
    Eigen::Matrix<double, 6, 6> Q = 400 * Eigen::MatrixXd::Identity(6, 6);
    Eigen::Matrix<double, 3, 3> R = 1 * Eigen::MatrixXd::Identity(3, 3);//WBC控制器的权重参数
    Eigen::VectorXd kp(nv), ki(nv), kd(nv);//WBC的PID参数
    double w_V = 0.0, w_qdd = 1.0;
    if (!FLAGS_real)//仿真
    {
      kp << 500, 500, 500, 500, 500, 500, Eigen::VectorXd::Constant(na, 500);
      ki << 0, 0, 0, 0, 0, 0, Eigen::VectorXd::Constant(na, 0);
      kd << 45, 45, 45, 45, 45, 45, Eigen::VectorXd::Constant(na, 45);
    }
    else//实物
    {
      kp << 20, 20, 20, 50, 0, 100, Eigen::VectorXd::Constant(na, 10);
      ki << 0, 0, 0, 0, 0, 0, Eigen::VectorXd::Constant(na, 0);
      kd << 0.5, 0.5, 0.5, 1, 0, 1, Eigen::VectorXd::Constant(na, 1);
    }
    auto wbc = builder.AddSystem<WholeBodyController>(plant, contacts_frame_name,
                                                      Q, R, kp, ki, kd, w_V, w_qdd);

    if (!FLAGS_real)//仿真
    {
      state_ext = builder.AddSystem<StateExt>(plant, end_frames_name);
      builder.Connect(plant->get_state_output_port(), state_ext->get_state_input_port());
      builder.Connect(state_ext->get_state_est_output_port(), trajectory->get_state_est_input_port());
      builder.Connect(trajectory->get_state_des_output_port(), wbc->get_state_des_input_port());
      builder.Connect(state_ext->get_state_est_output_port(), wbc->get_state_est_input_port());
      builder.Connect(wbc->get_tau_output_port(), plant->get_actuation_input_port());

      Eigen::Matrix<double, 6, 1> fDisturb;
      fDisturb << 0, 0, 0, 0, 0, 0;//仿真的干扰力
      auto forceDisturber = builder.AddSystem<ForceDisturber>(plant->GetBodyByName("base_link").index(), fDisturb, 1, 0.1, 3);
      builder.Connect(forceDisturber->get_output_port(), plant->get_applied_spatial_force_input_port());
    }
    else//实物
    {
      hard_plant = builder.AddSystem<HardwarePlant>(plant, end_frames_name, FLAGS_dt, 2);
      act_mul = builder.AddSystem<ActuationMultiplexer>(plant);
      builder.Connect(hard_plant->get_state_est_output_port(), trajectory->get_state_est_input_port());
      builder.Connect(trajectory->get_state_des_output_port(), wbc->get_state_des_input_port());
      builder.Connect(hard_plant->get_state_est_output_port(), wbc->get_state_est_input_port());
      builder.Connect(wbc->get_tau_output_port(), act_mul->get_tau_input_port());
      builder.Connect(trajectory->get_state_des_output_port(), hard_plant->get_state_des_input_port());
      builder.Connect(act_mul->get_output_port(), hard_plant->get_actuation_input_port());
    }

    systems::VectorLogSink<double> *logger;
    if (FLAGS_log == true)
      logger = LogVectorOutput(plant->get_state_output_port(), &builder);

    auto diagram = builder.Build();//建立框图
    std::unique_ptr<systems::Context<double>> diagram_context = diagram->CreateDefaultContext();

    systems::Context<double> &plant_context = diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
    initialState(plant, &plant_context, com_squat, end_frames_name, initial_joint_name, initial_joint_pos);//初始化状态

    Eigen::VectorXd q0 = plant->GetPositions(plant_context);

    systems::Context<double> &trajectory_context = diagram->GetMutableSubsystemContext(*trajectory, diagram_context.get());
    trajectory->Initialize(trajectory_context, q0);

    if (!FLAGS_real)//仿真
    {
      state_ext->Initialize(q0);
    }
    else//实物
    {
      uint32_t na = plant->num_actuated_dofs();
      Eigen::VectorXd tau(na);
      tau.setZero();
      plant->get_actuation_input_port().FixValue(&plant_context, tau);

      std::vector<double> q_initial_vec;
      q_initial_vec.assign(&q_initial[nq_f], q_initial.data() + q_initial.rows() * q_initial.cols());//没看懂呢？
      for (uint32_t i = 0; i < q_initial_vec.size(); i++)
      {
        q_initial_vec[i] *= (180.0 / M_PI);//弧度转度
      }
      jointMoveTo(q_initial_vec, 90, FLAGS_dt);

      systems::Context<double> &h_plant_context = diagram->GetMutableSubsystemContext(*hard_plant, diagram_context.get());
      hard_plant->Initialize(h_plant_context, q0);

      systems::Context<double> &act_mul_context = diagram->GetMutableSubsystemContext(*act_mul, diagram_context.get());
      // Eigen::VectorXd tau_max = Eigen::VectorXd::Constant(na, MAX_CURRENT);
      Eigen::VectorXd tau_max = Eigen::Vector3d(MAX_CURRENT, 60., MAX_CURRENT);
      act_mul->get_tau_max_input_port().FixValue(&act_mul_context, tau_max);

      FLAGS_realtime = 1.0;
    }

    std::cout << "Simulation ...\n";
    systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
    simulator.set_target_realtime_rate(FLAGS_realtime);//可以设置仿真速度
    if (FLAGS_log == true)
    {
      simulator.AdvanceTo(4);
    }
    else
    {
      simulator.AdvanceTo(std::numeric_limits<double>::infinity());
    }
    std::cout << "Simulation complete\n";

    if (FLAGS_log == true)
    {
      const auto &log = logger->FindLog(simulator.get_context());
      recordLog(plant, log);
    }

    return 0;
  }
} // namespace drake

void sigintHandler(int sig)
{
  printf("\n");
  HWPlantDeInit();
  printf("signal exit.\n");
  exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[])
{
  sched_process(0);//设置优先级
  gflags::ParseCommandLineFlags(&argc, &argv, true);//在main函数里面完成对gflags参数的初始，其中第三个参数为remove_flag，如果为true，gflags会移除parse过的参数，否则gflags就会保留这些参数，但可能会对参数顺序进行调整。
  if (!lc.good())//LCM不正常，返回-1
  {
    std::cout << "Error: Lcm not good!\n";
    return -1;
  }
  if (FLAGS_real)
  {
    signal(SIGINT, sigintHandler);
    if (HWPlantInit(lc) != 0)
    {
      return -2;
    }
  }
  foot_pos.setZero();//足端位置置0
  drake::doMain();//运行drake相关
  return 0;
}
