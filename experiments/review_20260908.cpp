// Focused efficiency probes; compile separately against the existing core.
#include "boussinesq_kernel.hpp"
#include "cheb_basis.hpp"
#include "h2_operator.hpp"
#include "contact_solver.hpp"
#include "fourier_precond.hpp"
#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <algorithm>
#include <omp.h>

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}

void table_free() {
    const int ns=1024, ls=8, q=6;
    const double h=1.0/ns, a=0.5*h, scale=1.0/M_PI;
    auto start=Clock::now();
    hmc::BoussinesqKernel kernel(ns, 1., 1.);
    double table_time=seconds(start);
    start=Clock::now();
    hmc::H2Operator A(kernel, {ls,q,1}); A.build();
    double build_time=seconds(start);
    // Only (near_radius+1)*ls offsets are ever requested. Preserve the
    // original arithmetic and parity to test bitwise equivalence.
    const int span=2*ls;
    start=Clock::now();
    std::vector<double> near(span*span);
    for(int y=0;y<span;++y) for(int x=0;x<span;++x)
        near[y*span+x]=scale*hmc::love_uz(x*h,y*h,a,a);
    hmc::H2Operator B(ns,h,
        [=](double x,double y){return scale*hmc::love_uz(x,y,a,a);},
        [&](int x,int y){return near[std::abs(y)*span+std::abs(x)];},
        {ls,q,1});
    B.build();
    double compact_time=seconds(start);
    Eigen::VectorXd x=Eigen::VectorXd::Random(ns*ns), ya,yb;
    A.matvec_into(x,ya); B.matvec_into(x,yb);
    std::printf("table_free ns=%d table=%.6f build=%.6f compact_build=%.6f rel=%.3e equal=%d table_bytes=%zu compact_bytes=%zu\n",
        ns,table_time,build_time,compact_time,(ya-yb).norm()/ya.norm(),
        (ya.array()==yb.array()).all(),std::size_t(ns)*ns*8,near.size()*8);
    auto info=A.info();
    // Lower bound, omitting capacities, leaves, transfer matrices, kernel,
    // per-thread temporaries, allocator overhead and optional float caches.
    auto omitted=info.n_boxes*sizeof(hmc::H2Box)+8*info.n_far_interactions+
        8*info.n_near_interactions+8*(info.n_boxes+1+info.n_leaves+1);
    std::printf("h2_memory reported=%lld omitted_metadata_lower_bound=%zu\n",
        (long long)info.bytes_total,omitted);
}

void transfer_bench() {
    for(int q:{4,6,8}) {
        hmc::ChebBasis basis(q);
        Eigen::MatrixXd A(q,q),B(q,q),R(q*q,q*q);
        for(int j=0;j<q;++j) {
            A.col(j)=basis.weights(-0.5+0.5*basis.nodes[j]);
            B.col(j)=basis.weights(0.5+0.5*basis.nodes[j]);
        }
        for(int ay=0;ay<q;++ay) for(int ax=0;ax<q;++ax)
            for(int by=0;by<q;++by) for(int bx=0;bx<q;++bx)
                R(ax+q*ay,bx+q*by)=A(ax,bx)*B(ay,by);
        const int n=50000;
        Eigen::MatrixXd x=Eigen::MatrixXd::Random(q*q,n),yd(q*q,n),yt(q*q,n);
        double td=1e9,tt=1e9;
        for(int rep=0;rep<5;++rep) {
            auto start=Clock::now();
            for(int i=0;i<n;++i) yd.col(i).noalias()=R*x.col(i);
            td=std::min(td,seconds(start));
            Eigen::MatrixXd tmp(q,q);
            start=Clock::now();
            for(int i=0;i<n;++i) {
                Eigen::Map<const Eigen::MatrixXd> X(x.col(i).data(),q,q);
                Eigen::Map<Eigen::MatrixXd> Y(yt.col(i).data(),q,q);
                tmp.noalias()=A*X; Y.noalias()=tmp*B.transpose();
            }
            tt=std::min(tt,seconds(start));
        }
        std::printf("transfer q=%d dense=%.6f tensor=%.6f speedup=%.3f rel=%.3e\n",
            q,td,tt,td/tt,(yd-yt).norm()/yd.norm());
    }
}

void preconditioner_scale() {
    const int ns=64, n=ns*ns;
    hmc::BoussinesqKernel K(ns,1.,1.);
    hmc::H2Operator A(K,{8,6,1}); A.build();
    hmc::FourierPreconditioner M(ns);
    Eigen::VectorXd g(n), reference;
    for(int iy=0;iy<ns;++iy) for(int ix=0;ix<ns;++ix) {
        const double x=(ix+0.5)/ns, y=(iy+0.5)/ns;
        g(iy*ns+ix)=0.02*(std::cos(2*M_PI*(x+2*y)+0.3)+
              0.45*std::cos(2*M_PI*(3*x+y)+1.1)+
              0.3*std::cos(2*M_PI*(2*x+4*y)+2.));
    }
    hmc::MatVec op=[&](const Eigen::VectorXd& x){return A.matvec(x);};
    for(double scale:{1.,1e-3,1e3}) {
        hmc::Precond pre=[&](const Eigen::VectorXd& x,const std::vector<std::uint8_t>& mask)->Eigen::VectorXd {
            return scale*M.apply(x,mask);
        };
        auto r=hmc::solve_contact(op,g,0.005,1e-10,1200,true,pre);
        if(scale==1.) reference=r.pressure;
        auto v=(A.matvec(r.pressure)+g).eval();
        double fw=r.pressure.dot(v)-r.pressure.sum()*v.minCoeff();
        std::printf("precond_scale factor=%.0e it=%d converged=%d error=%.3e fw=%.3e pressure_rel=%.3e\n",
            scale,r.iterations,r.converged,r.error,fw,(r.pressure-reference).norm()/reference.norm());
    }
}

int main() {
    std::printf("threads=%d\n",omp_get_max_threads());
    table_free(); transfer_bench(); preconditioner_scale();
}
